#!/usr/bin/env python3
"""Interactive terminal chat for Inkling-2-Mini on Apple Silicon with 2.06-bit SRFT KV compression."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

_omlx = os.environ.get("OMLX_PATH")
if _omlx and _omlx not in sys.path:
    sys.path.insert(0, str(Path(_omlx).resolve()))

import mlx.core as mx
from omlx.patches.inkling2.vlm import load
from src.m6_accel import install_m6_patches, make_m6_cache


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


def prefill_chunked(model, ids: list[int], cache, chunk: int = 1024):
    x = mx.array([ids], dtype=mx.int32)
    logits = None
    for s in range(0, len(ids), chunk):
        logits = model(x[:, s : min(len(ids), s + chunk)], cache=cache)
        mx.eval(logits)
    return logits


def main():
    ap = argparse.ArgumentParser(description="Inkling-2-Mini 1M-Context Apple Silicon Chat")
    ap.add_argument("--model", default="mlx-community/Inkling-2-Mini-oQ4e-mtp")
    ap.add_argument("--kv", choices=("lm2_srft", "bf16"), default="lm2_srft")
    ap.add_argument("--max-ctx", type=int, default=131072, help="Global KV capacity (up to 1048576)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--system", type=str, default="You are a helpful, concise assistant.")
    a = ap.parse_args()

    install_m6_patches()
    print(f"Loading {a.model} ({a.kv}, max_ctx={a.max_ctx:,})...", flush=True)
    t0 = time.time()
    model, proc = load(a.model)
    tok = proc.tokenizer
    eos_ids = {int(model.config.eos_token_id)}
    if getattr(tok, "eos_token_id", None) is not None:
        eos_ids.add(int(tok.eos_token_id))
    fmt = "bf16" if a.kv == "bf16" else "lm2"
    cache = make_m6_cache(model, fmt, global_capacity=a.max_ctx)
    print(f"Ready in {time.time()-t0:.1f}s | active={mx.get_active_memory()/1024**3:.2f} GiB")
    print("Commands: /reset, /stats, /quit\n")

    history: list[dict] = []
    cached_len = 0

    def encode_messages(msgs: list[dict]) -> list[int]:
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                return tok.encode(text, add_special_tokens=False)
            except Exception:
                pass
        return tok.encode("\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs) + "\n\nAssistant:")

    while True:
        try:
            user_msg = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_msg:
            continue
        if user_msg in ("/quit", "/exit"):
            print("Bye!")
            break
        if user_msg == "/reset":
            history.clear()
            cached_len = 0
            del cache
            mx.clear_cache()
            cache = make_m6_cache(model, fmt, global_capacity=a.max_ctx)
            print("[cache cleared]")
            continue
        if user_msg == "/stats":
            kv_gb = sum(c.caches[0].nbytes for c in cache) / (1024 ** 3)
            print(f"[stats] ctx={cached_len:,} tok | KV={kv_gb:.2f} GiB | active={mx.get_active_memory()/1024**3:.2f} GiB")
            continue

        if not history and a.system:
            history.append({"role": "system", "content": a.system})
        history.append({"role": "user", "content": user_msg})
        full_ids = encode_messages(history)
        new_ids = full_ids[cached_len:]
        if not new_ids:
            del cache
            mx.clear_cache()
            cache = make_m6_cache(model, fmt, global_capacity=a.max_ctx)
            new_ids = full_ids
            cached_len = 0

        t_pre0 = time.perf_counter()
        logits = prefill_chunked(model, new_ids, cache)
        t_pre = time.perf_counter() - t_pre0
        cached_len += len(new_ids)
        nxt = sample_token(logits, a.temp, a.top_p)
        mx.eval(nxt)

        print("\nInkling> ", end="", flush=True)
        gen_ids, prev_text = [], ""
        t_gen0 = time.perf_counter()
        for _ in range(a.max_tokens):
            tid = int(nxt[0, 0])
            if tid in eos_ids:
                break
            gen_ids.append(tid)
            cur_text = tok.decode(gen_ids, skip_special_tokens=True)
            if len(cur_text) > len(prev_text) and not cur_text.endswith("\ufffd"):
                print(cur_text[len(prev_text):], end="", flush=True)
                prev_text = cur_text
            logits = model(nxt, cache=cache)
            nxt = sample_token(logits, a.temp, a.top_p)
            mx.eval(nxt)
            cached_len += 1
        t_gen = time.perf_counter() - t_gen0
        final_text = tok.decode(gen_ids, skip_special_tokens=True)
        if len(final_text) > len(prev_text):
            print(final_text[len(prev_text):], end="", flush=True)
        print()
        history.append({"role": "assistant", "content": final_text})
        kv_gb = sum(c.caches[0].nbytes for c in cache) / (1024 ** 3)
        print(f"\n[{len(gen_ids)} tok @ {len(gen_ids)/max(t_gen,1e-6):.1f} tok/s | prefill {len(new_ids)} tok in {t_pre:.2f}s | ctx={cached_len:,} | KV={kv_gb:.2f} GiB]", flush=True)


if __name__ == "__main__":
    main()
