#!/usr/bin/env python3
"""OpenAI-compatible API server for Inkling-2-Mini on Apple Silicon (2.06-bit SRFT 1M KV).

Designed for OpenCode, Cursor, Continue, and any OpenAI-compatible client:
  - GET  /health
  - GET  /v1/models
  - POST /v1/chat/completions  (SSE streaming + non-streaming, reasoning_content, tool_calls)
  - Automatic KV prefix-cache reuse across multi-turn agent loops!

Usage:
  OMLX_PATH=./omlx python server.py --kv lm2_srft --max-ctx 262144 --port 8080
  OMLX_PATH=./omlx python server.py --kv lm2_srft --max-ctx 1048576 --port 8080
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_omlx = os.environ.get("OMLX_PATH")
if _omlx and _omlx not in sys.path:
    sys.path.insert(0, str(Path(_omlx).resolve()))

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from omlx.patches.inkling2.vlm import load
from src.m6_accel import install_m6_patches, make_m6_cache

# Inkling-2 special control markers
_RE_BLOCK = re.compile(
    r"<\|message_model\|>(?P<header>[^<]*)"
    r"<\|(?P<kind>content_thinking|content_text|content_invoke_tool_json)\|>"
    r"(?P<body>.*?)"
    r"(?:<\|end_message\|>|$)",
    re.DOTALL,
)


def parse_inkling_output(raw: str) -> tuple[str, str, list[dict]]:
    """Parse raw Inkling-2 generation into (reasoning_content, content, tool_calls)."""
    raw = raw.replace("<|content_model_end_sampling|>", "")
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    matches = list(_RE_BLOCK.finditer(raw))
    if not matches:
        # Fallback if model emitted plain text without header tags
        cleaned = re.sub(r"<\|[^|]+\|>", "", raw).strip()
        return "", cleaned, []

    for m in matches:
        kind = m.group("kind")
        body = m.group("body")
        header = m.group("header").strip()
        if kind == "content_thinking":
            reasoning_parts.append(body)
        elif kind == "content_text":
            text_parts.append(body)
        elif kind == "content_invoke_tool_json":
            try:
                obj = json.loads(body.strip())
                fn_name = obj.get("name") or header
                fn_args = obj.get("args", {})
                args_str = json.dumps(fn_args, ensure_ascii=False) if not isinstance(fn_args, str) else fn_args
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": fn_name, "arguments": args_str},
                })
            except Exception:
                text_parts.append(body)

    return "".join(reasoning_parts).strip(), "".join(text_parts).strip(), tool_calls


def canonicalize_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages into Inkling-2's Jinja chat_template contract."""
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("role") == "assistant" and m.get("tool_calls"):
            new_tcs = []
            for tc in m["tool_calls"]:
                tc_c = dict(tc)
                fn = dict(tc_c.get("function", {}))
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                fn["arguments"] = args
                tc_c["function"] = fn
                new_tcs.append(tc_c)
            m["tool_calls"] = new_tcs
        out.append(m)
    return out


def sample_token(logits: mx.array, temp: float, top_p: float) -> mx.array:
    if temp <= 0.0:
        return mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int32)
    lg = logits[0, -1, :].astype(mx.float32) / temp
    if 0.0 < top_p < 1.0:
        probs = mx.softmax(lg, axis=-1)
        idx = mx.argsort(-probs)
        sorted_p = probs[idx]
        cum_p = mx.cumsum(sorted_p)
        keep = mx.concatenate([mx.array([True]), (cum_p <= top_p)[:-1]])
        masked = mx.where(keep, sorted_p, 0.0)
        masked = masked / mx.sum(masked)
        pick = mx.random.categorical(mx.log(masked + 1e-12))
        return idx[pick].reshape(1, 1).astype(mx.int32)
    return mx.random.categorical(lg).reshape(1, 1).astype(mx.int32)


class Engine:
    """Single-worker-thread Inkling-2-Mini engine (keeps all MLX GPU streams on one OS thread)."""

    def __init__(self, model_id: str, kv_mode: str, max_ctx: int, default_effort: str):
        self.model_id = model_id
        self.kv_mode = kv_mode
        self.fmt = "bf16" if kv_mode == "bf16" else "lm2"
        self.max_ctx = max_ctx
        self.default_effort = default_effort
        self._exec = ThreadPoolExecutor(max_workers=1)
        self._exec.submit(self._load_on_worker).result()

    def _load_on_worker(self):
        install_m6_patches()
        t0 = time.time()
        print(f"[server] Loading {self.model_id} (kv={self.kv_mode}, max_ctx={self.max_ctx:,})...", flush=True)
        self.model, self.proc = load(self.model_id)
        self.tok = self.proc.tokenizer
        self.eos_ids = {int(self.model.config.eos_token_id)}
        if getattr(self.tok, "eos_token_id", None) is not None:
            self.eos_ids.add(int(self.tok.eos_token_id))
        self.cache = None
        self.cached_ids: list[int] = []
        print(f"[server] Ready in {time.time()-t0:.1f}s | active={mx.get_active_memory()/1024**3:.2f} GiB", flush=True)

    def _encode(self, messages: list[dict], tools: list[dict] | None, effort: str | None) -> list[int]:
        msgs = canonicalize_messages(messages)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "reasoning_effort": effort or self.default_effort,
        }
        if tools:
            kwargs["tools"] = tools
        text = self.tok.apply_chat_template(msgs, **kwargs)
        return self.tok.encode(text, add_special_tokens=False)

    def _prefill_with_prefix_reuse(self, prompt_ids: list[int]):
        # Reuse existing KV cache if the new prompt strictly extends cached_ids!
        reuse = (
            self.cache is not None
            and len(self.cached_ids) > 0
            and len(prompt_ids) >= len(self.cached_ids)
            and prompt_ids[: len(self.cached_ids)] == self.cached_ids
        )
        if reuse and len(prompt_ids) > len(self.cached_ids):
            delta = prompt_ids[len(self.cached_ids) :]
        else:
            self.cache = None
            self.cached_ids = []
            mx.clear_cache()
            self.cache = make_m6_cache(self.model, self.fmt, global_capacity=self.max_ctx)
            delta = prompt_ids

        x = mx.array([delta], dtype=mx.int32)
        logits = None
        chunk = 1024
        for s in range(0, len(delta), chunk):
            logits = self.model(x[:, s : min(len(delta), s + chunk)], cache=self.cache)
            mx.eval(logits)
        self.cached_ids = list(prompt_ids)
        return logits, len(prompt_ids) - len(delta)

    def generate(self, messages: list[dict], tools: list[dict] | None,
                 max_tokens: int, temperature: float, top_p: float,
                 effort: str | None, stop: list[str] | None = None):
        return self._exec.submit(
            self._generate_on_worker, messages, tools, max_tokens, temperature, top_p, effort, stop
        ).result()

    def _generate_on_worker(self, messages: list[dict], tools: list[dict] | None,
                            max_tokens: int, temperature: float, top_p: float,
                            effort: str | None, stop: list[str] | None = None):
        prompt_ids = self._encode(messages, tools, effort)
        t0 = time.perf_counter()
        logits, reused = self._prefill_with_prefix_reuse(prompt_ids)
        t_pre = time.perf_counter() - t0

        nxt = sample_token(logits, temperature, top_p)
        mx.eval(nxt)

        gen_ids: list[int] = []
        t_gen0 = time.perf_counter()
        finish_reason = "length"
        for _ in range(max_tokens):
            tid = int(nxt[0, 0])
            if tid in self.eos_ids:
                finish_reason = "stop"
                break
            gen_ids.append(tid)
            self.cached_ids.append(tid)
            logits = self.model(nxt, cache=self.cache)
            nxt = sample_token(logits, temperature, top_p)
            mx.eval(nxt)
        t_gen = time.perf_counter() - t_gen0

        raw = self.tok.decode(gen_ids, skip_special_tokens=False)
        reasoning, content, tool_calls = parse_inkling_output(raw)
        if tool_calls:
            finish_reason = "tool_calls"

        tps = len(gen_ids) / max(t_gen, 1e-6)
        kv_gb = sum(c.caches[0].nbytes for c in self.cache) / (1024 ** 3)
        print(f"[req] prompt={len(prompt_ids)} (reused={reused}, pre={t_pre:.2f}s) | "
              f"gen={len(gen_ids)} @ {tps:.1f} tok/s | tools={len(tool_calls)} | KV={kv_gb:.2f} GiB",
              flush=True)

        return {
            "reasoning": reasoning,
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(gen_ids),
        }


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "inkling-2-mini-1m"
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int | None = 4096
    max_completion_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    reasoning_effort: str | None = None


def create_app(engine: Engine, served_name: str) -> FastAPI:
    app = FastAPI(title="inkling-1m-mlx OpenAI Server")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": served_name, "kv_mode": engine.kv_mode, "max_ctx": engine.max_ctx}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": served_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "inkling-1m-mlx",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        max_toks = req.max_completion_tokens or req.max_tokens or 4096
        stop_list = [req.stop] if isinstance(req.stop, str) else req.stop
        try:
            res = engine.generate(
                messages=req.messages,
                tools=req.tools,
                max_tokens=max_toks,
                temperature=req.temperature,
                top_p=req.top_p,
                effort=req.reasoning_effort,
                stop=stop_list,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        usage = {
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "total_tokens": res["prompt_tokens"] + res["completion_tokens"],
        }

        if not req.stream:
            msg: dict[str, Any] = {"role": "assistant", "content": res["content"] or None}
            if res["reasoning"]:
                msg["reasoning_content"] = res["reasoning"]
            if res["tool_calls"]:
                msg["tool_calls"] = res["tool_calls"]
            return {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "message": msg, "finish_reason": res["finish_reason"]}],
                "usage": usage,
            }

        # OpenAI SSE streaming response
        def sse():
            def chunk(delta: dict, finish: str | None = None, inc_usage: bool = False):
                payload: dict[str, Any] = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                if inc_usage:
                    payload["usage"] = usage
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield chunk({"role": "assistant", "content": ""})
            if res["reasoning"]:
                yield chunk({"reasoning_content": res["reasoning"]})
            if res["content"]:
                yield chunk({"content": res["content"]})
            if res["tool_calls"]:
                for idx, tc in enumerate(res["tool_calls"]):
                    yield chunk({
                        "tool_calls": [{
                            "index": idx,
                            "id": tc["id"],
                            "type": "function",
                            "function": tc["function"],
                        }]
                    })
            yield chunk({}, finish=res["finish_reason"], inc_usage=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main():
    ap = argparse.ArgumentParser(description="OpenAI-compatible server for Inkling-2-Mini (2-bit SRFT 1M KV)")
    ap.add_argument("--model", default="mlx-community/Inkling-2-Mini-oQ4e-mtp")
    ap.add_argument("--served-name", default="inkling-2-mini-1m")
    ap.add_argument("--kv", choices=("lm2_srft", "bf16"), default="lm2_srft")
    ap.add_argument("--max-ctx", type=int, default=262144, help="Global KV capacity (up to 1048576)")
    ap.add_argument("--reasoning-effort", default="low",
                    choices=("none", "minimal", "low", "medium", "high"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    engine = Engine(a.model, a.kv, a.max_ctx, a.reasoning_effort)
    app = create_app(engine, a.served_name)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
