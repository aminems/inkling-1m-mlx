# inkling-1m-mlx

**One Million Tokens in 20 GB: Calibration-Free 2-Bit SRFT KV-Cache Compression for Looped MoE Inference on Apple Silicon**

Run Thinking Machines Lab's 26.4B looped MoE model (**Inkling-2-Mini**, 48 logical visits, 1M native context) at its **full 1,048,576-token context window in 19.49 GiB at 9.52 tok/s** on a **32 GB Apple M6 Mac mini**.

- **Paper:** [`paper/main.pdf`](paper/main.pdf)
- **Upstream oMLX Sliding-Cache Fix:** [jundot/omlx#5512](https://github.com/jundot/omlx/pull/5512)

---

## Highlights (32 GB Apple M6, `Inkling-2-Mini-oQ4e-mtp`)

| Context | Stock `omlx` | `m6_bf16` *(PR #5512)* | `m6_q4_srft` (4.5b) | **`m6_lm2_srft` (2.06b FFT)** |
|---:|---|---|---|---|
| **256K** | *OOM (~63 GiB)* | 5.09 tok/s (23.34 GiB) | 12.50 tok/s (17.62 GiB) | **17.09 tok/s · 1.27 GiB KV · 16.39 GiB peak (3.36× vs bf16)** |
| **512K** | *OOM (~112 GiB)* | *OOM (~31.4 GiB)* | 8.42 tok/s (19.87 GiB) | **13.81 tok/s · 2.31 GiB KV · 17.43 GiB peak** |
| **1,048,576 (1M)** | *OOM (~208 GiB)* | *OOM (~47.3 GiB)* | 3.66 tok/s (24.36 GiB) | **9.52 tok/s · 4.37 GiB KV · 19.49 GiB peak** |

### Quality (`4 × 4096` WikiText-2 & 8K–32K Needle-in-a-Haystack)
| Format | Eff. Bits | WT2 PPL | ΔPPL | 8K NIAH | 16K NIAH | 32K NIAH |
|---|---:|---:|---:|:---:|:---:|:---:|
| `bf16` (ref) | 16.000 | 6.7489 | +0.0000 | PASS | PASS | PASS |
| `SRFT lm3` | 3.0625 | 6.7418 | −0.0071 | PASS | PASS | PASS |
| `SRFT K3/V2` | 2.5625 | 6.7725 | +0.0236 | PASS | PASS | PASS |
| **`SRFT lm2`** | **2.0625** | **6.8785** | **+0.1296** | **PASS** | **PASS** | **PASS** |

---

## Key Techniques

1. **Scale-free 2.000-bit Value compression via SRFT (FFT):** Inkling-2 enforces parameter-free unit-RMS normalization on $V$ (`use_value_norm_no_gamma=True`). Rotating with an orthonormal Subsampled Randomized Fourier Transform (SRFT) turns $V$ into an isotropic unit-variance Gaussian that needs **zero per-token scales** under 2-bit Lloyd–Max quantization (**2.0625** eff. bits overall, $7.76\times$ smaller than bf16, zero offline calibration).
2. **Split banded-bias attention with exact online LSE merge:** Splits the 8 global visits into a quantized far prefix `[0 : N - 1024]` (where Inkling-2's 1,024-wide relative bias is identically zero) and an exact bf16 recent window `[N - 1024 : N]` + learned sink, merged via exact two-way online log-sum-exp.
3. **256-entry `half4` LUT GQA-packed Metal kernel (`src/kv_lm2_metal.py`):** Maps 4 coordinates (1 byte) per Metal simdgroup lane, decodes all 4 coordinates in a single 2 KiB threadgroup LUT lookup, serves all $r=3$ GQA query heads in registers with zero padding waste, and uses Block-8 lazy `exp2` rescaling (reaching 73.4 GB/s and 7.65 ms/layer at 1M tokens).

---

## Quickstart

```bash
# 1. Clone this repo and oMLX (for the inkling2_loop loader)
git clone https://github.com/aminems/inkling-1m-mlx.git
cd inkling-1m-mlx
git clone --depth 1 https://github.com/jundot/omlx.git

# 2. Install dependencies (Apple Silicon, Python >= 3.11)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Chat interactively with 1M context in < 20 GiB!
OMLX_PATH=./omlx python chat.py --kv lm2_srft --max-ctx 1048576
```

## Citation

```bibtex
@misc{bergach2026inkling1m,
  title  = {One Million Tokens in 20\,GB: Calibration-Free 2-Bit SRFT KV-Cache Compression for Looped MoE Inference on Apple Silicon},
  author = {Bergach, Mohamed Amine},
  year   = {2026},
  howpublished = {\url{https://github.com/aminems/inkling-1m-mlx}}
}
```

## License

MIT License — see [`LICENSE`](LICENSE).
