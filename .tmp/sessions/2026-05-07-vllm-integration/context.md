# Task Context: vLLM Integration for SDFT On-Policy Generation

Session ID: 2026-05-07-vllm-integration
Created: 2026-05-07
Status: in_progress

## Current Request
Integrate vLLM's synchronous LLM engine into the SDFT workflow to replace slow `model.generate()` for on-policy completions. Phase 1: Build SDFTVLLMEngine, wire it into sdft_plugin.py and workflow.py, add config fields.

## Context Files (Standards to Follow)
- /home/nimish/Programs/LlamaFactory/CLAUDE.md — Project architecture, code style (ruff, 119 char limit, double quotes, Google docstrings), commands
- /home/nimish/Programs/LlamaFactory/pyproject.toml — Ruff lint rules, formatting config, isort, Python 3.11+
- /home/nimish/Programs/LlamaFactory/Makefile — make style, make quality, make test
- /home/nimish/Programs/LlamaFactory/.pre-commit-config.yaml — Pre-commit hooks

## Reference Files (Source Material to Look At)
- src/llamafactory/train/sdft_plugin.py — SDFTTrainerWrapper, _generate_completions_on_policy, compute_loss
- src/llamafactory/train/sdft/workflow.py — run_sdft, base_trainer creation
- src/llamafactory/train/sdft/__init__.py — Module exports
- scripts/vllm_infer.py — Synchronous LLM usage pattern with SamplingParams
- src/llamafactory/chat/vllm_engine.py — Existing async VLLMEngine (reference, not to be reused)
- src/llamafactory/hparams/finetuning_args.py — SDFT config fields (alpha, beta, use_vllm_for_generation, etc.)
- src/llamafactory/model/loader.py — load_model() pattern
- src/llamafactory/train/sft/trainer.py — CustomSeq2SeqTrainer class
- src/llamafactory/train/trainer_utils.py — create_ref_model() pattern

## Components (Phase 1)
1. SDFTVLLMEngine class — wraps vLLM LLM, manages init/destroy/weight-sync lifecycle
2. Weight sync — periodic LoRA merge → temp save → engine recreate
3. _generate_completions_on_policy vLLM path — replace dead code with real sync LLM call
4. sdft/workflow.py init — create vLLM engine, register callback
5. Config — vllm_sync_every field, adjust existing fields

## Constraints
- Python 3.11+, Apache 2.0 license header in new files
- Must work with LoRA (merge_and_unload pattern)
- Must fallback gracefully to model.generate() when vLLM unavailable
- Target: 2×24GB GPUs (Qwen3.5-4B for testing)
- Sync every 16-32 steps to amortize engine recreation cost (~2s)

## Exit Criteria
- [ ] SDFTVLLMEngine class created and importable
- [ ] vLLM path in _generate_completions_on_policy replaced with working sync LLM call
- [ ] Weight sync callback registered in workflow.py
- [ ] vllm_sync_every config field added
- [ ] make style passes
- [ ] Manual smoke test with Qwen3.5-4B + identity dataset
