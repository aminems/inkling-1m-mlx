"""Standalone M6 acceleration module for Inkling-2-Mini (26.4B looped MoE):
  - SRFT: real orthonormal FFT rotation (MLX native Metal rfft/irfft)
  - SlidingWindowKVCache(1024): bounded chronological cache for the 40 sliding visits
  - GlobalLM2Cache: packed 2.06-bit SRFT Lloyd-Max (scale-free 2.000-bit V!) for the 8 global visits
  - Split banded-bias attention with exact online LSE merge
"""
from __future__ import annotations
import os, sys
from pathlib import Path

_omlx_path = os.environ.get("OMLX_PATH")
if _omlx_path and _omlx_path not in sys.path:
    sys.path.insert(0, str(Path(_omlx_path).resolve()))

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from omlx.patches.inkling2 import kernels as omlx_kernels
from omlx.patches.inkling2 import model as omlx_model

from .kv_lm2_metal import LM2Kernels


class SRFT:
    """Real orthonormal Subsampled Randomized Fourier Transform (d-point RFFT + random +/-1 signs)."""

    def __init__(self, d: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.signs = mx.array(rng.choice([-1.0, 1.0], size=(d,)).astype(np.float32))
        self.inv_sqrt_d = float(d ** -0.5)
        self.sqrt_2_over_d = float((2.0 / d) ** 0.5)

    def fwd(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        xs = x.astype(mx.float32) * self.signs
        F = mx.fft.rfft(xs, axis=-1)
        dc = F[..., 0:1].real * self.inv_sqrt_d
        nyq = F[..., -1:].real * self.inv_sqrt_d
        mid = F[..., 1:-1]
        out = mx.concatenate([dc, nyq, mid.real * self.sqrt_2_over_d, mid.imag * self.sqrt_2_over_d], axis=-1)
        return out.astype(dtype)

    def inv(self, y: mx.array) -> mx.array:
        dtype = y.dtype
        yf = y.astype(mx.float32)
        h = self.d // 2 - 1
        sqrt_d = float(self.d ** 0.5)
        inv_s2 = float((self.d / 2.0) ** 0.5)
        dc = yf[..., 0:1] * sqrt_d
        nyq = yf[..., 1:2] * sqrt_d
        re = yf[..., 2 : 2 + h] * inv_s2
        im = yf[..., 2 + h :] * inv_s2
        F = mx.concatenate([dc + 0j, re + 1j * im, nyq + 0j], axis=-1)
        xs = mx.fft.irfft(F, n=self.d, axis=-1)
        return (xs * self.signs).astype(dtype)


class SlidingWindowKVCache:
    """Bounded chronological sliding-window KV cache (amortized trim; never exceeds 2*window)."""

    def __init__(self, window: int = 1024, step: int = 256):
        self.window = window
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0
        self._buf_len = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, n_kv, L, D = keys.shape
        prev = self._buf_len
        if self.keys is None or (prev + L) > self.keys.shape[2]:
            if self.keys is not None and prev >= 2 * self.window:
                keep = self.window
                old_k = self.keys[:, :, prev - keep : prev, :]
                old_v = self.values[:, :, prev - keep : prev, :]
                new_cap = max(2 * self.window + self.step, keep + L)
                new_k = mx.zeros((B, n_kv, new_cap, D), dtype=keys.dtype)
                new_v = mx.zeros((B, n_kv, new_cap, D), dtype=values.dtype)
                new_k[:, :, :keep, :] = old_k
                new_v[:, :, :keep, :] = old_v
                self.keys, self.values = new_k, new_v
                prev = keep
            else:
                new_cap = ((prev + L + self.step - 1) // self.step) * self.step
                new_k = mx.zeros((B, n_kv, new_cap, D), dtype=keys.dtype)
                new_v = mx.zeros((B, n_kv, new_cap, D), dtype=values.dtype)
                if self.keys is not None and prev > 0:
                    new_k[:, :, :prev, :] = self.keys[:, :, :prev, :]
                    new_v[:, :, :prev, :] = self.values[:, :, :prev, :]
                self.keys, self.values = new_k, new_v

        self.keys[:, :, prev : prev + L, :] = keys
        self.values[:, :, prev : prev + L, :] = values
        self._buf_len = prev + L
        self.offset += L
        needed = min(self._buf_len, self.window + L - 1)
        start = self._buf_len - needed
        return self.keys[:, :, start : self._buf_len, :], self.values[:, :, start : self._buf_len, :]

    def trim(self, n: int) -> int:
        n = min(n, self._buf_len)
        self._buf_len -= n
        self.offset -= n
        return n

    @property
    def nbytes(self) -> int:
        return 0 if self.keys is None else (self.keys.nbytes + self.values.nbytes)


class GlobalLM2Cache:
    """Packed 2.0625-bit SRFT (FFT) Lloyd-Max global KV cache (scale-free 2.000-bit V!)."""

    def __init__(self, n_kv: int = 8, n_q: int = 24, d: int = 128,
                 recent: int = 1024, capacity: int = 65536, spill_chunk: int = 256, seed: int = 0):
        self.n_kv, self.n_q, self.d = n_kv, n_q, d
        self.recent = recent
        self.spill_chunk = spill_chunk
        self.srft_k = SRFT(d, seed=seed * 2)
        self.srft_v = SRFT(d, seed=seed * 2 + 1)
        self.kern = LM2Kernels()
        self.capacity = capacity
        self.kc = mx.zeros((n_kv, capacity, 32), dtype=mx.uint8)
        self.ks = mx.zeros((n_kv, capacity), dtype=mx.float16)
        self.vc = mx.zeros((n_kv, capacity, 32), dtype=mx.uint8)
        self.n_far = 0
        self.near_cap = recent + spill_chunk * 2
        self.near_k = None
        self.near_v = None
        self.n_near = 0
        self.offset = 0

    def _grow_far(self, needed: int):
        new_cap = max(self.capacity * 2, needed + 4096)
        nkc = mx.zeros((self.n_kv, new_cap, 32), dtype=mx.uint8)
        nks = mx.zeros((self.n_kv, new_cap), dtype=mx.float16)
        nvc = mx.zeros((self.n_kv, new_cap, 32), dtype=mx.uint8)
        if self.n_far > 0:
            nkc[:, :self.n_far, :] = self.kc[:, :self.n_far, :]
            nks[:, :self.n_far] = self.ks[:, :self.n_far]
            nvc[:, :self.n_far, :] = self.vc[:, :self.n_far, :]
        self.kc, self.ks, self.vc = nkc, nks, nvc
        self.capacity = new_cap

    def update(self, keys: mx.array, values: mx.array):
        B, n_kv, L, D = keys.shape
        if self.near_k is None or (self.n_near + L) > self.near_k.shape[2]:
            new_cap = max(self.near_cap, self.n_near + L + self.spill_chunk)
            nk = mx.zeros((B, n_kv, new_cap, D), dtype=keys.dtype)
            nv = mx.zeros((B, n_kv, new_cap, D), dtype=values.dtype)
            if self.near_k is not None and self.n_near > 0:
                nk[:, :, :self.n_near, :] = self.near_k[:, :, :self.n_near, :]
                nv[:, :, :self.n_near, :] = self.near_v[:, :, :self.n_near, :]
            self.near_k, self.near_v = nk, nv
        self.near_k[:, :, self.n_near : self.n_near + L, :] = keys
        self.near_v[:, :, self.n_near : self.n_near + L, :] = values
        self.n_near += L
        self.offset += L

        if self.n_near >= self.recent + self.spill_chunk:
            spill = self.n_near - self.recent
            if self.n_far + spill > self.capacity:
                self._grow_far(self.n_far + spill)
            sk = self.srft_k.fwd(self.near_k[:, :, :spill, :])
            sv = self.srft_v.fwd(self.near_v[:, :, :spill, :])
            kq, ksc, vq = self.kern.quantize_kv(sk, sv)
            self.kc[:, self.n_far : self.n_far + spill, :] = kq
            self.ks[:, self.n_far : self.n_far + spill] = ksc
            self.vc[:, self.n_far : self.n_far + spill, :] = vq
            self.n_far += spill
            rem = self.n_near - spill
            self.near_k[:, :, :rem, :] = self.near_k[:, :, spill : self.n_near, :]
            self.near_v[:, :, :rem, :] = self.near_v[:, :, spill : self.n_near, :]
            self.n_near = rem

    def fetch_near(self):
        return self.near_k[:, :, :self.n_near, :], self.near_v[:, :, :self.n_near, :]

    def fetch_full_dequant(self, dtype):
        kn, vn = self.fetch_near()
        if self.n_far == 0:
            return kn, vn
        kf, vf = self.kern.dequantize_kv(
            self.kc[:, :self.n_far, :], self.ks[:, :self.n_far], self.vc[:, :self.n_far, :], dtype
        )
        return mx.concatenate([self.srft_k.inv(kf), kn], axis=2), mx.concatenate([self.srft_v.inv(vf), vn], axis=2)

    @property
    def nbytes(self) -> int:
        b = self.kc.nbytes + self.ks.nbytes + self.vc.nbytes
        if self.near_k is not None:
            b += self.near_k.nbytes + self.near_v.nbytes
        return b


@mx.compile
def _near_window_with_rel_and_merge(q, k_near, v_near, rel, sink, out_far, lse_far, scale: float):
    nq = q.shape[1]
    nkv = k_near.shape[1]
    g = nq // nkv
    W = k_near.shape[2]
    qf = q.reshape(1, nkv, g, 1, -1).astype(mx.float32)
    kf = k_near[:, :, None, :, :].astype(mx.float32)
    scores = ((qf @ kf.swapaxes(-1, -2)) * scale).reshape(1, nq, 1, W)

    rel_t = rel.transpose(0, 2, 1, 3).astype(mx.float32)
    R = rel_t.shape[-1]
    if W <= R:
        scores = scores + rel_t[..., :W][..., ::-1]
    else:
        pad = mx.zeros((1, nq, 1, W - R), dtype=mx.float32)
        scores = scores + mx.concatenate([pad, rel_t[..., ::-1]], axis=-1)

    sink_f = sink.reshape(1, nq, 1, 1).astype(mx.float32)
    m_near = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink_f)
    p_near = mx.exp(scores - m_near)
    p_sink = mx.exp(sink_f - m_near)
    l_near = mx.sum(p_near, axis=-1, keepdims=True) + p_sink
    lse_near = m_near + mx.log(l_near)

    vf = v_near[:, :, None, :, :].astype(mx.float32)
    out_near = ((p_near / l_near).reshape(1, nkv, g, 1, W) @ vf).reshape(1, nq, 1, -1)

    m = mx.maximum(lse_far, lse_near)
    w_far = mx.exp(lse_far - m)
    w_near = mx.exp(lse_near - m)
    out = (w_far * out_far.astype(mx.float32) + w_near * out_near) / (w_far + w_near)
    return out.astype(q.dtype)


def make_m6_cache(model, global_fmt: str = "lm2", global_capacity: int = 131072):
    args = model.args.text
    caches = []
    for visit_idx, (idx, _loop_pass) in enumerate(model.model._schedule):
        if args.is_sliding(idx):
            kv = SlidingWindowKVCache(window=args.sliding_window_size)
        elif global_fmt == "bf16":
            kv = KVCache()
        else:
            kv = GlobalLM2Cache(
                n_kv=args.num_key_value_heads,
                n_q=args.num_attention_heads,
                d=args.head_dim,
                recent=args.rel_extent,
                capacity=global_capacity,
                seed=visit_idx,
            )
        caches.append(CacheList(kv, ArraysCache(4)))
    return caches


def install_m6_patches():
    orig_attend = omlx_model.Inkling2Attention.attend

    def patched_attend(self, q, k, v, rel, sink, kv):
        if isinstance(kv, GlobalLM2Cache):
            kv.update(k, v)
            if q.shape[0] == 1 and q.shape[2] == 1 and kv.n_far > 0:
                q_far = kv.srft_k.fwd(q)
                out_far, lse_far = kv.kern.decode_with_lse(q_far, kv.kc, kv.ks, kv.vc, kv.n_far, self.scale)
                out_far = kv.srft_v.inv(out_far)
                kn, vn = kv.fetch_near()
                return _near_window_with_rel_and_merge(
                    q, kn, vn, rel, sink, out_far, lse_far, self.scale
                )
            kf, vf = kv.fetch_full_dequant(q.dtype)
            return omlx_kernels.banded_sink_attention(
                q, kf, vf, rel, sink, scale=self.scale, window=self.window, left_padding=None
            )
        return orig_attend(self, q, k, v, rel, sink, kv)

    omlx_model.Inkling2Attention.attend = patched_attend
