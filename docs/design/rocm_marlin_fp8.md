# ROCm Marlin FP8 Backend (CDNA1/2/3)

## Motivation

FP8 support must not be limited to the newest AMD parts. The goal is to deliver
Marlin-style FP8 GEMM on CDNA1/CDNA2/CDNA3, with **mandatory emulation** for
older fleets so they remain useful for inference workloads.

This design mirrors the existing CUDA Marlin interfaces in vLLM while keeping
HIP changes minimal and professional, with a clean path to add int4/mxfp4/nvfp4
later.

## Scope (Phase 1)

- Provide `fp8_marlin_gemm` on ROCm with:
  - **Emulated FP8** on CDNA1/CDNA2 (gfx908/gfx90a).
  - **Native FP8 MFMA** on CDNA3 (gfx94x, gfx95x) when available.
- Keep the op signature and tensor layout identical to the CUDA Marlin path.
- No Python API churn; reuse the same `torch.ops._C.*` entrypoints.

## Requirements

- Emulated FP8 is **mandatory** and always available on CDNA1/CDNA2.
- Native FP8 is used where supported, but the API and accuracy semantics are
  consistent with emulation.
- Workspace semantics and scale handling match CUDA Marlin.
- Minimal build-system disruption and isolated HIP sources.

## Non-Goals (Phase 1)

- Full int4/mxfp4/nvfp4 kernels (these follow in Phase 2+).
- Kernel auto-tuning; we will use deterministic heuristics consistent with
  existing Marlin behavior.

## Architecture Overview

### File Layout

- `csrc/quantization/fp8/fp8_marlin_hip.cu`
  - HIP entrypoint and arch dispatch.
- `csrc/quantization/fp8/fp8_marlin_hip_kernel.hip`
  - FP8 emulation + native MFMA kernels.
- `csrc/quantization/marlin_hip/marlin_hip_common.h`
  - Shared permutes/layout helpers, scale handling, and arch utilities.

### Kernel Strategy

**Emulated FP8 (CDNA1/CDNA2)**  
- Decode FP8 (E4M3/E5M2 per vLLM convention) to FP16/BF16 in registers.
- Fuse scale during decode.
- MFMA on FP16/BF16 with FP32 accumulation.
- Use LDS double/triple buffering and wave-grouped producer/consumer pipeline
  (mirrors HipKittens patterns).

**Native FP8 (CDNA3)**  
- Use FP8 MFMA instructions if supported.
- Same layout/scale semantics as emulation.
- Prefer MFMA builtins; use rocWMMA where stable.

### Dispatch Rules

- CDNA3 (gfx94x/gfx95x): native FP8 MFMA path.
- CDNA1/2 (gfx908/gfx90a): emulated FP8 path.
- Other archs: return a clear "not supported" error or fallback to Triton path
  in Python (to be decided per vLLM policy).

## Build Integration

- Add HIP sources to `VLLM_EXT_SRC` when `VLLM_GPU_LANG == HIP`.
- Keep CUDA sources unchanged; this is a HIP-only addition.
- Use existing `VLLM_GPU_ARCHES` for gfx908/gfx90a/gfx94x/gfx95x.

## Python Integration

- Keep `fp8_marlin_gemm` op name unchanged.
- Enable ROCm usage based on device arch checks in
  `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`.

## Testing

- ROCm-only test coverage for FP8 Marlin:
  - correctness vs reference for emulation (CDNA1/2)
  - correctness vs native path (CDNA3)
- Kernel perf smoke tests using vLLM benchmarks on CDNA1/2/3.

## Roadmap

Phase 2+ reuses the same structure to add:
- int4 (W4A16) Marlin HIP kernels
- mxfp4 and nvfp4 kernels
- unified dispatch tables keyed by quant type
