# vLLM Integration for SDFT On-Policy Generation

> **Status**: Phase 1 (MVP) in progress
> **Session**: `.tmp/sessions/2026-05-07-vllm-integration/context.md`

## Overview

Replace the slow `model.generate()` path in `_generate_completions_on_policy()` with
vLLM's synchronous `LLM` engine for 10-50x faster on-policy sampling.

Currently generation dominates training time (>99% of each step). With `max_new_tokens=512`
at 80s/step, this is unsustainable for real training runs. vLLM brings each step under 0.5s.

---

## Architecture

```
Training Step (after integration, ~200ms)

 vLLM Gen        Student Fwd      Teacher Fwd     KL Loss + Backward    Optim Step
 (prompts)       (prompt+comp)    (demo+comp)
 ~50ms  ──────▶  ~30ms  ──────▶  ~30ms  ──────▶  ~40ms  ──────────▶  ~20ms

Every K=32 steps: LoRA merge → temp save → engine recreate (~2s, ~3% overhead)
```

## Weight Sync Strategy

vLLM's `LLM` has no built-in `update_weights()` API. We use **periodic merge-and-reload**:

```
Every sync_every steps:
  1. Ensure vLLM is idle (all generation requests completed)
  2. model.merge_and_unload() → merged state_dict (LoRA baked into base)
  3. torch.save(merged_state_dict, tmp_dir / "pytorch_model-merged.bin")
  4. Save config.json to tmp_dir
  5. del old_llm; gc.collect(); torch.cuda.empty_cache()
  6. llm = LLM(model=tmp_dir, ...)  ← new engine with updated weights
  7. Reload LoRA onto training model: PeftModel.from_pretrained(base_model, lora_path)
```

**Sync frequency**: `vllm_sync_every: 32` — frequent enough for near-on-policy generation,
infrequent enough that engine recreation overhead (~2s) is <5% of total runtime.

---

## Memory Budget (2x24GB GPUs)

| Component | GPU 0 (training) | GPU 1 (vLLM) |
|---|---|---|
| Model weights (8B FP16) | 16 GB (LoRA) | 16 GB |
| Optimizer states (LoRA 32-bit) | ~3 GB | — |
| vLLM KV cache | — | ~2 GB |
| Activations | ~2 GB | ~1 GB |
| **Total** | **~21 GB** | **~19 GB** |

Config requirements: `cutoff_len=1024-2048`, `max_new_tokens=128-256`,
`vllm_gpu_memory_utilization=0.85`.

---

## Implementation Plan

### Phase 1 (MVP) — Basic sync LLM integration

- [x] Plan documented in this file
- [ ] **1.1** `SDFTVLLMEngine` class — `src/llamafactory/train/sdft/vllm_engine.py`
  - [ ] 1.1a: `__init__()` — loads vLLM `LLM` from model path, configures `SamplingParams`
  - [ ] 1.1b: `generate(prompts: list[str]) -> list[str]` — synchronous batch generation
  - [ ] 1.1c: `sync_weights(model)` — merge LoRA → save to tmpdir → recreate LLM
  - [ ] 1.1d: `shutdown()` — delete LLM, free GPU memory
  - [ ] 1.1e: Apache 2.0 license header
- [ ] **1.2** Wire vLLM path into `_generate_completions_on_policy()`
  - [ ] 1.2a: Replace dead code (lines 334-349) with real `self.vllm_engine.generate()`
  - [ ] 1.2b: Pass `SamplingParams` matching `temperature`, `top_p`, `max_new_tokens` from config
  - [ ] 1.2c: Tokenize generated text → token IDs (preserve current return format)
  - [ ] 1.2d: Graceful fallback to `model.generate()` if engine not initialized
- [ ] **1.3** Wire into `sdft/workflow.py` (`run_sdft()`)
  - [ ] 1.3a: Import `SDFTVLLMEngine`
  - [ ] 1.3b: Create engine after model load: `vllm_engine = SDFTVLLMEngine(model, tokenizer, model_args, sdft_args)`
  - [ ] 1.3c: Pass engine to `SDFTTrainerWrapper(vllm_engine=vllm_engine, ...)`
  - [ ] 1.3d: Call `vllm_engine.shutdown()` in finally/cleanup
- [ ] **1.4** Config additions — `src/llamafactory/hparams/finetuning_args.py`
  - [ ] 1.4a: Add `vllm_sync_every: int = 32` field
  - [ ] 1.4b: Add `vllm_gpu_memory_utilization: float = 0.85` field
- [ ] **1.5** `SDFTDataCollator` — (unchanged, verify compatibility)
- [ ] **1.6** Verify with Qwen3.5-4B + identity dataset

### Phase 2 (Optimize) — Memory and speed tuning

- [ ] **2.1** GPU placement — pin training to GPU0, vLLM to GPU1 via `CUDA_VISIBLE_DEVICES` splits
- [ ] **2.2** Tune `gpu_memory_utilization` per model size
- [ ] **2.3** Auto-reduce `max_model_len` from `cutoff_len + max_new_tokens`
- [ ] **2.4** Profile: measure sync overhead vs generation speedup
- [ ] **2.5** Async generation: submit vLLM requests, do other work, collect results

### Phase 3 (Robustness) — Edge cases and cleanup

- [ ] **3.1** OOM recovery: if vLLM init fails with OOM, fall back to `model.generate()`
- [ ] **3.2** Multi-GPU tensor parallelism (`tensor_parallel_size > 1`)
- [ ] **3.3** Integration test with Ternary-Bonsai-8B + glaive_toolcall_100k
- [ ] **3.4** Pre-commit pass: `make style`, `make license`
- [ ] **3.5** Update example config `sdft_toolcall_config.yaml` with vLLM fields

### Phase 4 (Stretch) — Direct weight push

- [ ] **4.1** Investigate vLLM worker model access for direct weight copy
- [ ] **4.2** Eliminate engine recreation overhead (<50ms sync vs 2s)
- [ ] **4.3** Support teacher model weight sync to vLLM

---

## Key Files

| File | Role |
|---|---|
| `src/llamafactory/train/sdft/vllm_engine.py` | NEW — SDFTVLLMEngine class |
| `src/llamafactory/train/sdft_plugin.py` | MODIFY — `_generate_completions_on_policy()`, `__init__()` |
| `src/llamafactory/train/sdft/workflow.py` | MODIFY — engine init/teardown |
| `src/llamafactory/hparams/finetuning_args.py` | MODIFY — new config fields |

## Reference Implementations

- `scripts/vllm_infer.py` — Synchronous `LLM` + `SamplingParams` usage
- TRL `DistilTrainer` — Weight sync via merge-and-reload pattern
- `src/llamafactory/chat/vllm_engine.py` — Existing async engine (NOT reused, different API)

## Commands

```bash
# Run style checks
make style

# Run all tests
make test

# Manual smoke test (Phase 1)
CUDA_VISIBLE_DEVICES=0 uv run llmf train examples/sdft/sdft_config.yaml
```
