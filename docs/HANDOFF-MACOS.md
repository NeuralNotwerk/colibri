# Handoff: building and running colibrì locally on a Mac

Instructions-to-self (Claude) for continuing this work on a macOS machine.
Written 2026-07-17 on the Linux prod box (going offline — office overheating).
State of the work: branch `local-patches` on the fork `NeuralNotwerk/colibri`.

## Context you are inheriting

- `local-patches` = upstream `JustVugg/colibri` main (merged 2026-07-17) + commit
  `e03bd86`: all fp8-KV-review fixes and the vectorized CUDA fp8 kernels
  (15.8× prefill, 4–6× decode — measured on RTX 5090, see the commit message).
- `backup/local-patches-pre-sync-20260717` = pre-merge snapshot, for diffing.
- Prod deployment recipe (Linux, not for the Mac): `docker-compose.colibri-cuda.
  glm-5.2-744b-int4.gpu-0-1-2-3-4.yml` + `Dockerfile.colibri-cuda` in
  `/storage/inference_servers/` on the old box — the compose env comments are a
  goldmine of tuning history; they were NOT copied into this repo.
- Open feature request: `issue_turboquant.md` (repo root — read it next).

## Build (CPU, works on any Mac)

```sh
git clone git@github.com:NeuralNotwerk/colibri.git && cd colibri/c
git checkout local-patches
brew install libomp        # Apple clang has no OpenMP runtime; Makefile
                           # auto-detects Homebrew libomp (see Makefile ~L41)
make glm                   # plain CPU build; NEON kernels are baseline on arm64
make test-c                # MUST be green: includes exhaustive e4m3 roundtrip,
                           # kv_disk v1/v2 round-trip, kv_alloc f32+KV8
```

Notes:
- No `-march` needed; arm64 NEON activates automatically. `ARCH=native` opt-in
  appends `-mcpu=native` (byte-identical default build without it).
- CUDA does not exist here: everything you touched in `backend_cuda.cu` is
  compiled out. The CPU KV8 path (`kv_fp8.h`, `attention_rows`) is fully live.

## Build (Metal GPU, Apple Silicon)

```sh
make glm METAL=1           # opt-in Apple-GPU backend (errors on non-macOS)
make metal-test            # parity tests for backend_metal.mm
```

**KV8 is auto-disabled under COLI_METAL** (the Metal fused attention reads f32
rows — verified in the claims ledger). So on a Mac you can exercise:
- KV8 **CPU** attention (build without METAL, `KV8=1`) — the quant/dequant hot
  path, disk v2 format, all the review fixes;
- f32 Metal attention (METAL build) — but not fp8-on-GPU. Porting the
  vectorized fp8 kernels to Metal is untouched follow-up work.

## Run

The real model is GLM-5.2 744B int4 (~370 GB of experts + dense). It runs
fully-resident on a 512 GB Mac Studio (M3/M4 Ultra) at best; on anything
smaller, experts stream from SSD (the engine's original design — it works, just
slower; the per-layer LRU + OS page cache handle it).

```sh
# server (matches prod flags minus CUDA):
KV8=1 CTX=262144 DRAFT=2 python3 c/openai_server.py \
    --host 0.0.0.0 --port 8000 --max-tokens 262143 \
    --model <path-to-snapshot>
# quick chat / one-shot without the server: see c/coli and README "Useful knobs"
```

- Model fetch: `c/download_fp8.py` (HF/ModelScope) then
  `c/tools/convert_fp8_to_int4.py`, or pull the ready int4 snapshot
  `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp` from HF.
- `.coli_usage` / `.coli_kv` / `stats.txt` are written INTO the snapshot dir.
  Copy `.coli_usage` from the old box if you want the learned pin profile
  (23.4M routings of history as of 2026-07-17); it's `<snapshot>/.coli_usage`.
- `PIN=auto PIN_FREEZE=1` = frozen placement + stats accumulation (prod policy).

## Fast dev loop without the 744B model

- `make test-c` covers the KV8 core with no model at all.
- Tiny real-weights fixture: `c/tools/make_glm_bench_model.py` and
  `make_glm_oracle.py` build a small snapshot + golden outputs; the "tiny
  matrix" (32/32 f32, 30/32 fp8, DSA_FORCE CPU==CUDA) is the quality gate this
  repo's commits cite. On the Mac the CUDA half of that matrix is N/A.
- `c/tools/quant_ablation.py` + `eval_glm.py` for quality A/Bs (used for the
  "log-lik deltas inside int4 noise" KV8 claim — reuse for TurboQuant).

## Gotchas learned the hard way (don't re-learn these)

1. `fsync` not `fdatasync` in `kv_disk_append` — macOS has no fdatasync; also
   consider `fcntl(F_FULLFSYNC)` on macOS if you care about true power-loss
   safety (plain fsync doesn't flush the drive cache there).
2. `MLOCK` auto-defaults ON on macOS (`mem_should_wire`, glm.c ~5735) because
   the macOS memory compressor silently compresses idle pinned experts and
   turns "RAM hits" into decompression stalls. Leave it on.
3. OMP on Apple clang: if builds fail with `-fopenmp`, it's a missing/blown
   Homebrew libomp, not the code.
4. The OMP hot-team autotuning in glm.c is skipped when COLI_CUDA is set — not
   relevant on Mac, but if you port that guard, remember explicit `OMP_*` env
   always wins.
5. The prefill KV producer loop is now `omp parallel for` (glm.c ~2507) with
   the `kv_dev_valid` shrink hoisted OUT (serial min-pass above it). If you
   touch it, keep the shrink outside the parallel region.
