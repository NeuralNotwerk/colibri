# Feature request: TurboQuant (3–4 bit KV cache) as the next tier below KV8

Source: Google Research, "TurboQuant: redefining AI efficiency with extreme
compression" — https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
Related papers to fetch on the new machine: QJL (arXiv 2406.03482, "1-Bit
Quantized JL Transform for KV Cache Quantization with Zero Overhead") and the
PolarQuant/TurboQuant paper itself.

## What it is (from the blog)

Two-stage, data-oblivious KV-cache quantization, no training/calibration:

1. **PolarQuant** — random rotation of each vector, then polar coordinates
   (radius + angles) quantized on a fixed grid; most bits go to the primary
   signal. Rotation makes coordinates near-Gaussian so the grid is
   distribution-free.
2. **QJL residual** — Johnson–Lindenstrauss transform of the residual error,
   kept as 1 sign bit per component, with an asymmetric estimator (
   high-precision query × 1-bit data) at score time.

Claims: **3-bit KV with no measured accuracy loss** (Llama-3.1-8B, Gemma,
Mistral; LongBench / NIAH / RULER / ZeroSCROLLS clean), 6× KV memory vs f16,
**8× faster attention-logit computation at 4-bit vs f32 keys on H100**,
near-zero preprocessing, negligible runtime overhead, near-optimal distortion
bounds.

## Why it fits colibrì specifically

KV8 (fp8 e4m3 + per-row f32 scale) is in and validated: 584 B/token/layer-row
vs 2304 f32, quality inside int4 noise. TurboQuant is the same shape of idea
one rung down the ladder, and the plumbing KV8 built is exactly what it needs
(per-row sidecar metadata, `.coli_kv` versioned records, quant-on-append /
dequant-in-kernel, CPU/CUDA parity tests, the fp8 device shadow):

| tier | bytes/token/layer-row (K=512+R=64) | 256k latent pool | 4×1M slots |
|------|------------------------------------|------------------|------------|
| f32  | 2304                               | 47.7 GB          | 763 GB (impossible) |
| KV8  | 584                                | 12.1 GB          | 193 GB (barely) |
| TQ4  | ~296 + rot metadata                | ~6.2 GB          | ~99 GB |
| TQ3  | ~224 + rot metadata                | ~4.7 GB          | ~75 GB |

The second prize is the **DSA indexer cache `Ic`**: it is still f32 today (the
one uncompressed KV component — ~0.9 GB at 256k, ~3.6 GB at 1M, and it's pure
inner-product top-k search, i.e. literally the vector-search use case
TurboQuant was also built for). QJL alone on `Ic` may be the cheapest first win.

## Implementation sketch (mapped onto the existing code)

**The rotation trick that makes this cheap at attention time.** Dot products
are rotation-invariant, and rotation is linear. So:
- store rows in ROTATED quantized space: `enc(x) = Q(R·x)`;
- at score time rotate the query once per (layer, query): `score_t = (R·q) ·
  deq(row_t)` — O(d log d) once with a structured rotation (randomized
  Hadamard: sign flips + FWHT), NOT per history row;
- for the context accumulation, `Σ_t w_t · x_t = R⁻¹(Σ_t w_t · deq(row_t))` —
  accumulate in rotated space (exactly what the `cl[K]` accumulator in every
  absorb kernel already does), apply ONE inverse rotation per (head, token) at
  the end, before the `wv`/o_proj projection.
  → The kernels keep their structure: quantized-byte loads + FMA in the T-loop,
  plus one O(K log K) epilogue. The vectorized uint4/uchar4 load pattern from
  e03bd86 carries over unchanged (nibbles instead of bytes).

**Where each piece goes:**
- `c/kv_tq.h` (new, sibling of `kv_fp8.h`): host encode/decode. Per row:
  sign-flip vector seed + FWHT (power-of-2 dims: K=512 ✓, R=64 ✓), polar
  encode of the rotated vector at 3 or 4 bits/dim, per-row radius scale f32
  (reuse the existing `Lsc/Rsc` sidecar arrays as-is).
- `attention_rows` producer (glm.c ~2517): third branch next to the `g_kv8`
  one. The rmsnorm/rope staging is unchanged. Keep the producer loop parallel.
- CPU consumer (glm.c ~2796): `coli_tq_dequant_row` staging to f32, same shape
  as `coli_kv8_dequant_row`.
- CUDA: `tq` twins of the absorb/batch/stream/split/sel kernels. Nibble/3-bit
  unpack replaces `fp8_e4m3()`; the polar decode is a small LUT (angle grid) —
  put it in shared or constant memory. The device shadow (`kv_dev_sync8`) and
  the new pinned-async upload path work byte-for-byte (smaller rows, same
  layout logic). Grid/smem math identical.
- `.coli_kv` v3: new magic, record = rotated-quantized rows + radius scales +
  rotation seed(s) in the header. v1/v2→v3 upgrade follows the existing
  quantize-on-resume + rewrite-at-first-append pattern (kv_disk_load dt
  switch). Sanitize-on-load applies (bound-check angle codes).
- Budget: `kv_pool_bytes` gets the third bytes/row case; everything downstream
  (cap_for_ram, expert_avail, shadow projection) already keys off it.
- Env: `KV_TQ=3|4` (mutually exclusive with `KV8=1`, or `KV8=tq3` style — pick
  one, document in ENVIRONMENT.md).

**Phasing (each phase independently shippable + testable):**
1. `kv_tq.h` + exhaustive host tests (mirror `test_kv_fp8.c`: roundtrip
   distortion bounds, RNE/tie behavior of the angle grid, zero/NaN/Inf rows,
   FWHT self-inverse property). CPU attention path + disk v3 + budget. Gate:
   tiny-oracle matrix ≥ KV8's 30/32, real-model log-lik A/B inside int4 noise
   (reuse `quant_ablation.py` / `eval_glm.py` harness).
2. QJL on the DSA indexer `Ic` (f32 → ~1–2 bits/dim + query-side estimator in
   the selection scan, glm.c ~2589). Gate: selection overlap vs f32 top-k, and
   DSA_FORCE generation identical (selection must stay faithful when it
   selects everything).
3. CUDA kernels + shadow. Gate: cuda-test parity vs CPU reference at 1e-3 rms
   (same fixtures as kvdev8), bench vs KV8 kernels (expect ≥ parity: half the
   bytes through the same L1-bound loops).
4. Optional: QJL residual stage on the latent rows (the +1 bit that buys the
   3-bit quality claim), asymmetric estimator folded into the score loop.

## Open questions / risks (answer before phase 1 lands)

- **MLA regime is harsher than the paper's.** The compressed latent is both K
  and V ("l'errore entra negli score E nel context" — kv_fp8.h header), and
  DeepSeek-V3-class models only validated fp8 there. TurboQuant's results are
  on standard per-head K/V at 8B–9B scale. 3-bit may not survive the absorbed
  latent; 4-bit + QJL residual is the realistic target. Measure, don't assume.
- **Rope sub-vector (R=64):** position-encoded, high dynamic range; KV8 keeps
  it at 8-bit today. Maybe leave rope at fp8 and TQ only the K=512 latent
  (mixed-precision rows — the sidecar layout allows it trivially).
- **The absorbed context epilogue** needs `R⁻¹` per (head, token) — confirm the
  O(K log K) FWHT epilogue is negligible vs the T-loop at small T too (decode
  T-loop is only `nsel`≤4096 long on the DSA path).
- **Radius quantization:** per-row amax scale already exists; polar radius may
  want log-scale bins — check the paper.
- **License/code:** check whether Google released reference code and under
  what license before porting anything verbatim; otherwise implement from the
  papers (QJL has public code per its arXiv page).
- **Blackwell fp4:** sm_120 has native fp4 tensor cores — a hardware-fp4 KV
  tier is a *competing* simpler design for the CUDA path (no rotation, keep
  the exact KV8 kernel structure at half the bytes). Worth a bake-off in
  phase 3 before committing to TQ-on-GPU.

## Prior art already in-repo to lean on

- `kv_fp8.h` — the template for the whole feature.
- `e03bd86` — vectorized load patterns + parallel softmax the TQ kernels
  should inherit, and the pinned async shadow upload.
- Claims-ledger findings (2026-07-17 session): disk sanitize-on-load, truncate
  nrec clamp, MTP straddle row — all apply identically to a v3 format.
- `convert_fp8_to_int4.py` grouped-quant tooling — the cold-expert int2/grouped
  RAM tier idea pairs with this (attention and experts are the two RAM walls).
