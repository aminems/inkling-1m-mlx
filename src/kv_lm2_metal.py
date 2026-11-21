"""Packed 2-bit SRFT (FFT) Lloyd-Max GQA-packed Metal KV kernel for Inkling-2-Mini (Apple M6).

Memory layout (d = 128, 2 bits/coord = 32 bytes/row):
  - K codes:  (n_kv, cap, 32) uint8   [4 coords packed per byte: lane l owns coords 4*l..4*l+3]
  - K scales: (n_kv, cap)     float16 [1 per-token RMS scale -> 2.125 bits/coord]
  - V codes:  (n_kv, cap, 32) uint8   [Inkling-2 unit-RMS V -> ZERO scale array -> 2.000 bits/coord!]
  Total = 2.0625 effective bits/coord (7.76x smaller than bf16 -> 4.125 GiB for 1M context!).

Kernel design:
  - 1 simdgroup (32 lanes) per threadgroup; lane l owns 4 coords (1 byte load per token ->
    perfectly coalesced 32-byte burst per token).
  - Dequantizes K and V ONCE via a 4-entry register Lloyd-Max LUT and serves all r=3 GQA
    query heads in registers with ZERO padding waste and ZERO shared memory in the inner loop.
  - Split-K across threadgroups, followed by exact online LSE reduction.
"""
from __future__ import annotations
import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

# 2-bit unit-Gaussian Lloyd-Max centroids and midpoints
_LM2_C = (-1.5104176, -0.4527800, 0.4527800, 1.5104176)
_LM2_B = (-0.9815988, 0.0, 0.9815988)

_QUANT_LM2_SRC = r"""
    // Grid: (n_rows * 32, 1, 1), threadgroup: (128, 1, 1) -> 1 simdgroup (32 lanes) per row
    const uint row = thread_position_in_grid.x >> 5;
    const uint lane = thread_position_in_grid.x & 31u;
    const uint n_rows = params[0];
    if (row >= n_rows) return;

    const uint base = row * 128u + lane * 4u;
    // --- K: per-token RMS scale + 2-bit Lloyd-Max ---
    float k0 = float(k_in[base + 0u]);
    float k1 = float(k_in[base + 1u]);
    float k2 = float(k_in[base + 2u]);
    float k3 = float(k_in[base + 3u]);
    float ksq = k0*k0 + k1*k1 + k2*k2 + k3*k3;
    float k_rms = sqrt(simd_sum(ksq) * (1.0f / 128.0f) + 1e-12f);
    half k_rms_h = half(k_rms);
    float inv_k = 1.0f / float(k_rms_h);
    if (lane == 0u) k_scale[row] = k_rms_h;

    auto q2 = [](float v) -> uchar {
        uchar c = (v >= -0.9815988f) ? 1u : 0u;
        c += (v >= 0.0f) ? 1u : 0u;
        c += (v >= 0.9815988f) ? 1u : 0u;
        return c;
    };
    uchar kb = q2(k0 * inv_k) | (q2(k1 * inv_k) << 2u) | (q2(k2 * inv_k) << 4u) | (q2(k3 * inv_k) << 6u);
    k_codes[row * 32u + lane] = kb;

    // --- V: scale-free 2-bit Lloyd-Max (Inkling-2 unit-RMS V after orthonormal SRFT) ---
    float v0 = float(v_in[base + 0u]);
    float v1 = float(v_in[base + 1u]);
    float v2 = float(v_in[base + 2u]);
    float v3 = float(v_in[base + 3u]);
    uchar vb = q2(v0) | (q2(v1) << 2u) | (q2(v2) << 4u) | (q2(v3) << 6u);
    v_codes[row * 32u + lane] = vb;
"""

_DEQUANT_LM2_SRC = r"""
    const uint row = thread_position_in_grid.x >> 5;
    const uint lane = thread_position_in_grid.x & 31u;
    const uint n_rows = params[0];
    if (row >= n_rows) return;
    const float C[4] = {-1.5104176f, -0.4527800f, 0.4527800f, 1.5104176f};
    const uint base = row * 128u + lane * 4u;
    const uchar kb = k_codes[row * 32u + lane];
    const float ks = float(k_scale[row]);
    k_out[base + 0u] = T(C[kb & 3u] * ks);
    k_out[base + 1u] = T(C[(kb >> 2u) & 3u] * ks);
    k_out[base + 2u] = T(C[(kb >> 4u) & 3u] * ks);
    k_out[base + 3u] = T(C[(kb >> 6u) & 3u] * ks);

    const uchar vb = v_codes[row * 32u + lane];
    v_out[base + 0u] = T(C[vb & 3u]);
    v_out[base + 1u] = T(C[(vb >> 2u) & 3u]);
    v_out[base + 2u] = T(C[(vb >> 4u) & 3u]);
    v_out[base + 3u] = T(C[(vb >> 6u) & 3u]);
"""

_ATTN_LM2_SPLIT_SRC = r"""
    // Grid: (n_tgs * 32 * NSG, n_kv, 1), threadgroup: (32 * NSG, 1, 1)
    // Ultra-fast: 256-entry threadgroup half4 LUT (decodes 4 coords in 1 lookup!) + Block-8 exp2()
    threadgroup half4 LUT[256];
    {
        const ushort tid = ushort(thread_index_in_threadgroup);
        const ushort nthr = ushort(threads_per_threadgroup.x);
        const half C[4] = {half(-1.5104176f), half(-0.4527800f), half(0.4527800f), half(1.5104176f)};
        for (ushort idx = tid; idx < 256u; idx += nthr) {
            LUT[idx] = half4(C[idx & 3u], C[(idx >> 2u) & 3u], C[(idx >> 4u) & 3u], C[(idx >> 6u) & 3u]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kvh = threadgroup_position_in_grid.y;
    const uint part = threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint N = params[0];
    const uint cap = params[1];
    const uint chunk = params[2];
    const uint nparts = params[3];
    const float qscale2 = as_type<float>(params[4]);

    const uint t0 = part * chunk;
    const uint t1 = min(N, t0 + chunk);

    const uint qb0 = (kvh * 3u + 0u) * 128u + lane * 4u;
    const uint qb1 = (kvh * 3u + 1u) * 128u + lane * 4u;
    const uint qb2 = (kvh * 3u + 2u) * 128u + lane * 4u;
    const half4 q0 = half4(float(q_in[qb0])*qscale2, float(q_in[qb0+1u])*qscale2, float(q_in[qb0+2u])*qscale2, float(q_in[qb0+3u])*qscale2);
    const half4 q1 = half4(float(q_in[qb1])*qscale2, float(q_in[qb1+1u])*qscale2, float(q_in[qb1+2u])*qscale2, float(q_in[qb1+3u])*qscale2);
    const half4 q2 = half4(float(q_in[qb2])*qscale2, float(q_in[qb2+1u])*qscale2, float(q_in[qb2+2u])*qscale2, float(q_in[qb2+3u])*qscale2);

    device const uchar* kp = k_codes + ((ulong)kvh * cap) * 32ul + lane;
    device const half*  ks = k_scale + ((ulong)kvh * cap);
    device const uchar* vp = v_codes + ((ulong)kvh * cap) * 32ul + lane;

    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f;
    float l0 = 0.0f,   l1 = 0.0f,   l2 = 0.0f;
    float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f);

    uint t = t0;
    for (; t + 8u <= t1; t += 8u) {
        float s0[8], s1[8], s2[8];
        float bm0 = -1e30f, bm1 = -1e30f, bm2 = -1e30f;
        #pragma unroll
        for (uint j = 0u; j < 8u; ++j) {
            const half4 kv = LUT[kp[(ulong)(t + j) * 32ul]];
            const float sc = float(ks[t + j]);
            const float v_s0 = simd_sum(float(dot(q0, kv))) * sc;
            const float v_s1 = simd_sum(float(dot(q1, kv))) * sc;
            const float v_s2 = simd_sum(float(dot(q2, kv))) * sc;
            s0[j] = v_s0; bm0 = max(bm0, v_s0);
            s1[j] = v_s1; bm1 = max(bm1, v_s1);
            s2[j] = v_s2; bm2 = max(bm2, v_s2);
        }
        const float nm0 = max(m0, bm0); const float a0 = exp2(m0 - nm0); l0 *= a0; m0 = nm0; o0 *= a0;
        const float nm1 = max(m1, bm1); const float a1 = exp2(m1 - nm1); l1 *= a1; m1 = nm1; o1 *= a1;
        const float nm2 = max(m2, bm2); const float a2 = exp2(m2 - nm2); l2 *= a2; m2 = nm2; o2 *= a2;

        #pragma unroll
        for (uint j = 0u; j < 8u; ++j) {
            const float w0 = exp2(s0[j] - m0); l0 += w0;
            const float w1 = exp2(s1[j] - m1); l1 += w1;
            const float w2 = exp2(s2[j] - m2); l2 += w2;
            const float4 vv = float4(LUT[vp[(ulong)(t + j) * 32ul]]);
            o0 = fma(float4(w0), vv, o0);
            o1 = fma(float4(w1), vv, o1);
            o2 = fma(float4(w2), vv, o2);
        }
    }
    for (; t < t1; ++t) {
        const half4 kv = LUT[kp[(ulong)t * 32ul]];
        const float sc = float(ks[t]);
        const float s0 = simd_sum(float(dot(q0, kv))) * sc;
        const float s1 = simd_sum(float(dot(q1, kv))) * sc;
        const float s2 = simd_sum(float(dot(q2, kv))) * sc;
        const float4 vv = float4(LUT[vp[(ulong)t * 32ul]]);

        float nm0 = max(m0, s0), a0 = exp2(m0 - nm0), w0 = exp2(s0 - nm0); l0 = l0*a0 + w0; m0 = nm0; o0 = fma(float4(w0), vv, o0*a0);
        float nm1 = max(m1, s1), a1 = exp2(m1 - nm1), w1 = exp2(s1 - nm1); l1 = l1*a1 + w1; m1 = nm1; o1 = fma(float4(w1), vv, o1*a1);
        float nm2 = max(m2, s2), a2 = exp2(m2 - nm2), w2 = exp2(s2 - nm2); l2 = l2*a2 + w2; m2 = nm2; o2 = fma(float4(w2), vv, o2*a2);
    }

    const ulong obase = ((ulong)kvh * nparts + part) * 3ul;
    const ulong p0 = (obase + 0ul) * 128ul + lane * 4ul;
    const ulong p1 = (obase + 1ul) * 128ul + lane * 4ul;
    const ulong p2 = (obase + 2ul) * 128ul + lane * 4ul;
    opart[p0 + 0u] = o0.x; opart[p0 + 1u] = o0.y; opart[p0 + 2u] = o0.z; opart[p0 + 3u] = o0.w;
    opart[p1 + 0u] = o1.x; opart[p1 + 1u] = o1.y; opart[p1 + 2u] = o1.z; opart[p1 + 3u] = o1.w;
    opart[p2 + 0u] = o2.x; opart[p2 + 1u] = o2.y; opart[p2 + 2u] = o2.z; opart[p2 + 3u] = o2.w;
    if (lane == 0u) {
        const ulong mbase = ((ulong)kvh * nparts + part) * 6ul;
        mlpart[mbase + 0ul] = m0; mlpart[mbase + 1ul] = l0;
        mlpart[mbase + 2ul] = m1; mlpart[mbase + 3ul] = l1;
        mlpart[mbase + 4ul] = m2; mlpart[mbase + 5ul] = l2;
    }
"""

_REDUCE_LM2_SRC = r"""
    // Grid: (n_q * 128, 1, 1), threadgroup: (128, 1, 1)
    const uint qh = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    const uint nparts = params[3];
    const uint kvh = qh / 3u;
    const uint j = qh - kvh * 3u;

    float M = -1e30f;
    for (uint p = 0u; p < nparts; ++p) {
        const ulong mbase = ((ulong)kvh * nparts + p) * 6ul + (ulong)j * 2ul;
        const float lp = mlpart[mbase + 1ul];
        if (lp > 0.0f) M = max(M, mlpart[mbase]);
    }
    float L = 0.0f, acc = 0.0f;
    for (uint p = 0u; p < nparts; ++p) {
        const ulong mbase = ((ulong)kvh * nparts + p) * 6ul + (ulong)j * 2ul;
        const float lp = mlpart[mbase + 1ul];
        if (lp > 0.0f) {
            const float w = exp2(mlpart[mbase] - M);
            L += w * lp;
            const ulong obase = (((ulong)kvh * nparts + p) * 3ul + j) * 128ul + tid;
            acc += w * opart[obase];
        }
    }
    out[qh * 128u + tid] = TO(acc / max(L, 1e-30f));
    if (tid == 0u) {
        // Convert base-2 LSE back to natural-log LSE for merge with near window
        lse[qh] = (M + log2(max(L, 1e-30f))) * 0.6931471805599453f;
    }
"""


class LM2Kernels:
    """Packed 2-bit SRFT Lloyd-Max Metal kernels for Inkling-2-Mini (n_kv=8, r=3, d=128)."""

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
            cls._inst._init()
        return cls._inst

    def _init(self):
        self.nsg = 4
        opts = {"math_mode": "fast"}
        self.k_quant = mx.fast.metal_kernel(
            name="inkling_lm2_quant",
            input_names=["k_in", "v_in", "params"],
            output_names=["k_codes", "k_scale", "v_codes"],
            source=_QUANT_LM2_SRC,
            compile_options=opts,
        )
        self.k_dequant = mx.fast.metal_kernel(
            name="inkling_lm2_dequant",
            input_names=["k_codes", "k_scale", "v_codes", "params"],
            output_names=["k_out", "v_out"],
            source=_DEQUANT_LM2_SRC,
            compile_options=opts,
        )
        self.k_attn = mx.fast.metal_kernel(
            name="inkling_lm2_attn_split",
            input_names=["q_in", "k_codes", "k_scale", "v_codes", "params"],
            output_names=["opart", "mlpart"],
            source=_ATTN_LM2_SPLIT_SRC,
            header=f"#define NSG {self.nsg}\n",
            ensure_row_contiguous=False,
            compile_options=opts,
        )
        self.k_reduce = mx.fast.metal_kernel(
            name="inkling_lm2_reduce",
            input_names=["opart", "mlpart", "params"],
            output_names=["out", "lse"],
            source=_REDUCE_LM2_SRC,
            compile_options=opts,
        )

    def quantize_kv(self, k: mx.array, v: mx.array):
        """k, v: (1, n_kv, L, 128) -> k_codes (n_kv, L, 32) u8, k_scale (n_kv, L) f16, v_codes (n_kv, L, 32) u8."""
        _, nkv, L, d = k.shape
        rows = nkv * L
        params = mx.array([rows], dtype=mx.uint32)
        kc, ks, vc = self.k_quant(
            inputs=[k.reshape(rows * d), v.reshape(rows * d), params],
            grid=(rows * 32, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows * 32,), (rows,), (rows * 32,)],
            output_dtypes=[mx.uint8, mx.float16, mx.uint8],
        )
        return kc.reshape(nkv, L, 32), ks.reshape(nkv, L), vc.reshape(nkv, L, 32)

    def dequantize_kv(self, kc: mx.array, ks: mx.array, vc: mx.array, dtype):
        nkv, L, _ = kc.shape
        rows = nkv * L
        params = mx.array([rows], dtype=mx.uint32)
        ko, vo = self.k_dequant(
            inputs=[kc.reshape(rows * 32), ks.reshape(rows), vc.reshape(rows * 32), params],
            template=[("T", dtype)],
            grid=(rows * 32, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows * 128,), (rows * 128,)],
            output_dtypes=[dtype, dtype],
        )
        return ko.reshape(1, nkv, L, 128), vo.reshape(1, nkv, L, 128)

    def decode_with_lse(self, q: mx.array, kc: mx.array, ks: mx.array, vc: mx.array, N: int, scale: float):
        """q: (1, 24, 1, 128) -> out: (1, 24, 1, 128), lse: (1, 24, 1, 1)."""
        nkv, cap = kc.shape[0], kc.shape[1]
        nq = q.shape[1]
        # Target ~256 parts per KV head at >= 256K for full M6 GPU occupancy
        target_parts = 256 if N >= 262144 else (128 if N >= 65536 else 64)
        n_tgs = max(1, min(target_parts // self.nsg, (N + 128 - 1) // 128))
        nparts = n_tgs * self.nsg
        chunk = (N + nparts - 1) // nparts
        # Pre-multiply by log2(e) for hardware exp2()!
        qs = np.float32(scale * 1.4426950408889634).view(np.uint32)
        params = mx.array([N, cap, chunk, nparts, int(qs)], dtype=mx.uint32)

        opart, mlpart = self.k_attn(
            inputs=[q.reshape(nq * 128), kc, ks, vc, params],
            grid=(n_tgs * 32 * self.nsg, nkv, 1),
            threadgroup=(32 * self.nsg, 1, 1),
            output_shapes=[(nkv * nparts * 3 * 128,), (nkv * nparts * 6,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        out, lse = self.k_reduce(
            inputs=[opart, mlpart, params],
            template=[("TO", q.dtype)],
            grid=(nq * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(nq * 128,), (nq,)],
            output_dtypes=[q.dtype, mx.float32],
        )
        return out.reshape(1, nq, 1, 128), lse.reshape(1, nq, 1, 1)
