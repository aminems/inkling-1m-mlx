#!/usr/bin/env python3
"""OpenAI-compatible API server for Inkling-2-Mini on Apple Silicon (2.06-bit SRFT 1M KV).

Designed for OpenCode, Cursor, Continue, and any OpenAI-compatible client:
  - GET  /health
  - GET  /v1/models
  - POST /v1/chat/completions  (true token-by-token SSE streaming + non-streaming,
                                reasoning_content, two-phase OpenAI tool_calls)
  - Automatic KV prefix-cache reuse (protected from eviction by tiny background requests!)

Usage:
  OMLX_PATH=./omlx python server.py --kv lm2_srft --max-ctx 262144 --port 8080
  OMLX_PATH=./omlx python server.py --kv lm2_srft --max-ctx 1048576 --port 8080
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

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


def _parse_tool_json(body: str, header: str) -> dict | None:
    try:
        obj = json.loads(body.strip())
        fn_name = (obj.get("name") or header or "").strip()
        if not fn_name:
            return None
        fn_args = obj.get("args", obj.get("arguments", {}))
        args_str = json.dumps(fn_args, ensure_ascii=False) if not isinstance(fn_args, str) else fn_args
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": fn_name, "arguments": args_str},
        }
    except Exception:
        return None


def parse_inkling_output(raw: str) -> tuple[str, str, list[dict]]:
    """Parse raw Inkling-2 generation into (reasoning_content, content, tool_calls)."""
    raw = raw.replace("<|content_model_end_sampling|>", "")
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    matches = list(_RE_BLOCK.finditer(raw))
    if not matches:
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
            tc = _parse_tool_json(body, header)
            if tc is not None:
                tool_calls.append(tc)
            else:
                text_parts.append(body)

    return "".join(reasoning_parts).strip(), "".join(text_parts).strip(), tool_calls


def _flatten_content(content: Any) -> str:
    """Flatten OpenAI multi-part content arrays ([{'type':'text','text':'...'}]) into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in ("text", "input_text", "output_text") and "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
        return "\n".join(parts)
    return str(content)


def canonicalize_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages into Inkling-2's Jinja chat_template contract."""
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        if "content" in m and m["content"] is not None:
            m["content"] = _flatten_content(m["content"])
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


class _InklingStreamParser:
    """Incremental Inkling-2 special-token state machine for live token-by-token SSE."""

    def __init__(self, tok, emit: Callable[[str, Any], None]):
        self.tok = tok
        self.emit = emit
        self.state = "idle"  # idle | header | thinking | text | tool
        self.header_ids: list[int] = []
        self.block_ids: list[int] = []
        self.block_emitted = ""
        self.tool_calls: list[dict] = []

    def _flush_tool(self):
        body = self.tok.decode(self.block_ids, skip_special_tokens=True)
        header = self.tok.decode(self.header_ids, skip_special_tokens=True).strip()
        tc = _parse_tool_json(body, header)
        if tc is not None:
            self.tool_calls.append(tc)
            self.emit("tool_call", tc)
        elif body.strip():
            self.emit("content", body)

    def feed(self, tid: int):
        piece = self.tok.decode([tid], skip_special_tokens=False)
        if piece.startswith("<|") and piece.endswith("|>"):
            tag = piece[2:-2]
            if tag == "message_model":
                if self.state == "tool":
                    self._flush_tool()
                self.state = "header"
                self.header_ids = []
                self.block_ids = []
                self.block_emitted = ""
            elif tag == "content_thinking":
                self.state = "thinking"
                self.block_ids = []
                self.block_emitted = ""
            elif tag == "content_text":
                self.state = "text"
                self.block_ids = []
                self.block_emitted = ""
            elif tag == "content_invoke_tool_json":
                self.state = "tool"
                self.block_ids = []
                self.block_emitted = ""
            elif tag in ("end_message", "content_model_end_sampling"):
                if self.state == "tool":
                    self._flush_tool()
                self.state = "idle"
                self.header_ids = []
                self.block_ids = []
                self.block_emitted = ""
            return

        if self.state == "header":
            self.header_ids.append(tid)
        elif self.state == "tool":
            self.block_ids.append(tid)
        else:
            # thinking, text, or idle (fallback to content)
            self.block_ids.append(tid)
            full = self.tok.decode(self.block_ids, skip_special_tokens=True)
            # Avoid emitting incomplete UTF-8 replacement character at chunk boundary
            if full.endswith("\ufffd"):
                return
            delta = full[len(self.block_emitted):]
            if delta:
                self.block_emitted = full
                kind = "reasoning" if self.state == "thinking" else "content"
                self.emit(kind, delta)

    def finish(self):
        if self.state == "tool" and self.block_ids:
            self._flush_tool()
            self.state = "idle"
            self.block_ids = []


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

    def _prefill_with_prefix_reuse(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        on_progress: Callable[[int, int], None] | None = None,
    ):
        # Protect a large resident conversation KV cache from tiny background requests
        # (e.g. OpenCode's ~500-token session-title generator).
        reuse = (
            self.cache is not None
            and len(self.cached_ids) > 0
            and len(prompt_ids) >= len(self.cached_ids)
            and prompt_ids[: len(self.cached_ids)] == self.cached_ids
        )
        ephemeral = (
            not reuse
            and self.cache is not None
            and len(self.cached_ids) >= 2048
            and len(prompt_ids) < 2048
        )

        if ephemeral:
            cap = min(self.max_ctx, max(4096, len(prompt_ids) + max_tokens + 64))
            active_cache = make_m6_cache(self.model, self.fmt, global_capacity=cap)
            delta = prompt_ids
            reused = 0
        elif reuse and len(prompt_ids) > len(self.cached_ids):
            active_cache = self.cache
            delta = prompt_ids[len(self.cached_ids) :]
            reused = len(self.cached_ids)
        else:
            self.cache = None
            self.cached_ids = []
            mx.clear_cache()
            self.cache = make_m6_cache(self.model, self.fmt, global_capacity=self.max_ctx)
            active_cache = self.cache
            delta = prompt_ids
            reused = 0

        x = mx.array([delta], dtype=mx.int32)
        logits = None
        chunk = 1024
        for s in range(0, len(delta), chunk):
            e = min(len(delta), s + chunk)
            logits = self.model(x[:, s:e], cache=active_cache)
            mx.eval(logits)
            if on_progress is not None:
                on_progress(reused + e, len(prompt_ids))

        if not ephemeral:
            self.cached_ids = list(prompt_ids)
        return logits, active_cache, reused, ephemeral

    def _run_on_worker(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        effort: str | None,
        stream: bool,
        emit: Callable[[str, Any], None] | None = None,
    ) -> dict[str, Any]:
        prompt_ids = self._encode(messages, tools, effort)
        t0 = time.perf_counter()

        def on_prog(done: int, total: int):
            if emit is not None:
                emit("keepalive", f"prefill {done}/{total}")

        logits, active_cache, reused, ephemeral = self._prefill_with_prefix_reuse(
            prompt_ids, max_tokens, on_progress=on_prog
        )
        t_pre = time.perf_counter() - t0

        nxt = sample_token(logits, temperature, top_p)
        mx.eval(nxt)

        parser = _InklingStreamParser(self.tok, emit) if emit is not None else None
        gen_ids: list[int] = []
        t_gen0 = time.perf_counter()
        finish_reason = "length"
        for _ in range(max_tokens):
            tid = int(nxt[0, 0])
            if tid in self.eos_ids:
                finish_reason = "stop"
                break
            gen_ids.append(tid)
            if not ephemeral:
                self.cached_ids.append(tid)
            if parser is not None:
                parser.feed(tid)
            logits = self.model(nxt, cache=active_cache)
            nxt = sample_token(logits, temperature, top_p)
            mx.eval(nxt)
        t_gen = time.perf_counter() - t_gen0

        if parser is not None:
            parser.finish()

        raw = self.tok.decode(gen_ids, skip_special_tokens=False)
        reasoning, content, tool_calls = parse_inkling_output(raw)
        if parser is not None:
            tool_calls = parser.tool_calls
        if tool_calls:
            finish_reason = "tool_calls"

        tps = len(gen_ids) / max(t_gen, 1e-6)
        kv_gb = sum(c.caches[0].nbytes for c in active_cache) / (1024 ** 3)
        tc_names = [t["function"]["name"] for t in tool_calls]
        tag = " (ephemeral)" if ephemeral else ""
        print(
            f"[req] stream={stream} prompt={len(prompt_ids)} (reused={reused}, pre={t_pre:.2f}s){tag} | "
            f"gen={len(gen_ids)} @ {tps:.1f} tok/s | finish={finish_reason} | "
            f"r={len(reasoning)}c txt={len(content)}c tools={tc_names} | KV={kv_gb:.2f} GiB",
            flush=True,
        )

        if ephemeral:
            del active_cache
            mx.clear_cache()

        return {
            "reasoning": reasoning,
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(gen_ids),
        }

    async def generate_async(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        effort: str | None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._exec,
            self._run_on_worker,
            messages,
            tools,
            max_tokens,
            temperature,
            top_p,
            effort,
            False,
            None,
        )

    def start_stream(
        self,
        loop: asyncio.AbstractEventLoop,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        effort: str | None,
    ) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()

        def emit(ev: str, data: Any):
            loop.call_soon_threadsafe(q.put_nowait, (ev, data))

        def worker():
            try:
                res = self._run_on_worker(
                    messages, tools, max_tokens, temperature, top_p, effort, True, emit
                )
                emit("done", res)
            except Exception as e:
                emit("error", str(e))

        self._exec.submit(worker)
        return q


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
        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        if not req.stream:
            try:
                res = await engine.generate_async(
                    messages=req.messages,
                    tools=req.tools,
                    max_tokens=max_toks,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    effort=req.reasoning_effort,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            usage = {
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "total_tokens": res["prompt_tokens"] + res["completion_tokens"],
            }
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

        # Live token-by-token OpenAI SSE streaming (returns HTTP 200 immediately!)
        loop = asyncio.get_running_loop()
        q = engine.start_stream(
            loop=loop,
            messages=req.messages,
            tools=req.tools,
            max_tokens=max_toks,
            temperature=req.temperature,
            top_p=req.top_p,
            effort=req.reasoning_effort,
        )

        async def sse():
            def chunk(delta: dict, finish: str | None = None, usage: dict | None = None):
                payload: dict[str, Any] = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                if usage is not None:
                    payload["usage"] = usage
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield chunk({"role": "assistant", "content": ""})
            tc_index = 0
            while True:
                ev, data = await q.get()
                if ev == "keepalive":
                    yield f": {data}\n\n"
                elif ev == "reasoning":
                    yield chunk({"reasoning_content": data})
                elif ev == "content":
                    yield chunk({"content": data})
                elif ev == "tool_call":
                    # Standard two-phase OpenAI SSE tool_calls chunks (required by @ai-sdk/openai-compatible)
                    yield chunk({
                        "tool_calls": [{
                            "index": tc_index,
                            "id": data["id"],
                            "type": "function",
                            "function": {"name": data["function"]["name"], "arguments": ""},
                        }]
                    })
                    yield chunk({
                        "tool_calls": [{
                            "index": tc_index,
                            "function": {"arguments": data["function"]["arguments"]},
                        }]
                    })
                    tc_index += 1
                elif ev == "done":
                    usage = {
                        "prompt_tokens": data["prompt_tokens"],
                        "completion_tokens": data["completion_tokens"],
                        "total_tokens": data["prompt_tokens"] + data["completion_tokens"],
                    }
                    yield chunk({}, finish=data["finish_reason"], usage=usage)
                    yield "data: [DONE]\n\n"
                    break
                elif ev == "error":
                    yield f"data: {json.dumps({'error': {'message': data, 'type': 'server_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    break

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
