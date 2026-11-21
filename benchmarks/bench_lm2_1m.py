"""Reproduce the 256K / 512K / 1M-context packed 2.06-bit SRFT benchmark on Apple Silicon."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_omlx = os.environ.get("OMLX_PATH")
if _omlx and _omlx not in sys.path:
    sys.path.insert(0, str(Path(_omlx).resolve()))

import mlx.core as mx
from omlx.patches.inkling2.vlm import load
from src.kv_lm2_metal import LM2Kernels
from src.m6_accel import install_m6_patches, make_m6_cache, GlobalLM2Cache, SlidingWindowKVCache

SCALE = 128 ** -0.5


def fill_lm2_lowmem(cache, ctx_len: int):
    mx.random.seed(0)
    k_win = (mx.random.normal((1, 8, 1024, 128)) * 0.5).astype(mx.bfloat16)
    v_raw = mx.random.normal((1, 8, 1024, 128))
    v_win = (v_raw / mx.sqrt(mx.mean(v_raw * v_raw, axis=-1, keepdims=True))).astype(mx.bfloat16)
    mx.eval(k_win, v_win)

    for c in cache:
        kv = c.caches[0]
        if isinstance(kv, SlidingWindowKVCache):
            kv.update_and_fetch(k_win, v_win)
            kv.offset = ctx_len
            mx.eval(kv.keys, kv.values)
        elif isinstance(kv, GlobalLM2Cache):
            n_far = ctx_len - 1024
            sk, sv = kv.srft_k.fwd(k_win), kv.srft_v.fwd(v_win)
            kq, ksc, vq = kv.kern.quantize_kv(sk, sv)
            mx.eval(kq, ksc, vq)
            blk, bls, blv = mx.tile(kq, (1, 64, 1)), mx.tile(ksc, (1, 64)), mx.tile(vq, (1, 64, 1))
            mx.eval(blk, bls, blv)
            pos = 0
            while pos < n_far:
                take = min(65536, n_far - pos)
                kv.kc[:, pos : pos + take, :] = blk[:, :take, :]
                kv.ks[:, pos : pos + take] = bls[:, :take]
                kv.vc[:, pos : pos + take, :] = blv[:, :take, :]
                mx.eval(kv.kc, kv.ks, kv.vc)
                pos += take
            del blk, bls, blv
            kv.n_far = n_far
            kv.near_k = mx.zeros((1, 8, kv.near_cap, 128), dtype=mx.bfloat16)
            kv.near_v = mx.zeros((1, 8, kv.near_cap, 128), dtype=mx.bfloat16)
            kv.near_k[:, :, :1024, :] = k_win
            kv.near_v[:, :, :1024, :] = v_win
            kv.n_near = 1024
            kv.offset = ctx_len
            mx.eval(kv.near_k, kv.near_v)
            mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Inkling-2-Mini-oQ4e-mtp")
    ap.add_argument("--contexts", type=int, nargs="+", default=[262144, 524288, 1048576])
    a = ap.parse_args()

    install_m6_patches()
    model, _ = load(a.model)
    print(f"{'Ctx':>9} | {'Decode tok/s':>12} | {'ms/tok':>8} | {'KV GiB':>7} | {'Peak GiB':>8}")
    print("-" * 56)
    for ctx_len in a.contexts:
        cache = make_m6_cache(model, "lm2", global_capacity=ctx_len + 128)
        fill_lm2_lowmem(cache, ctx_len)
        nxt = mx.array([[1000]], dtype=mx.int32)
        for _ in range(3):
            logits = model(nxt, cache=cache)
            nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int32)
            mx.eval(nxt)
        mx.reset_peak_memory()
        gen = 20
        t0 = time.perf_counter()
        for _ in range(gen):
            logits = model(nxt, cache=cache)
            nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int32)
            mx.eval(nxt)
        dt = time.perf_counter() - t0
        kv_gb = sum(c.caches[0].nbytes for c in cache) / (1024 ** 3)
        peak_gb = mx.get_peak_memory() / (1024 ** 3)
        print(f"{ctx_len:>9d} | {gen/dt:>12.2f} | {(dt/gen)*1000:>8.2f} | {kv_gb:>7.2f} | {peak_gb:>8.2f}")
        del cache
        mx.clear_cache()


if __name__ == "__main__":
    main()
