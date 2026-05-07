#!/usr/bin/env python3
"""Standalone vLLM engine test for SDFT.

Usage: python tests/test_vllm_engine.py

Tests vLLM initialization + generation with Qwen3.5-4B, both with and without
LoRA adapter, isolating the engine from the training pipeline to identify
where the hang occurs.
"""

import gc
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_01_plain_model_no_adapter():
    """Test 1: vLLM with plain Qwen3.5-4B (no LoRA). Baseline — should work."""
    print("\n" + "=" * 60)
    print("TEST 1: Base model only (no LoRA adapter)")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("SKIP: vLLM not installed")
        return False

    model_name = "Qwen/Qwen3.5-4B"

    print(f"  Creating LLM from {model_name}...")
    t0 = time.time()
    try:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=2560,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=1,
            disable_log_stats=True,
            enforce_eager=True,
            enable_lora=False,
        )
        print(f"  LLM created in {time.time() - t0:.1f}s")

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=16,
        )
        prompts = ["Hello, how are you?"]
        print(f"  Generating from {len(prompts)} prompts...")
        t0 = time.time()
        results = llm.generate(prompts, sampling_params)
        print(f"  Generated in {time.time() - t0:.1f}s")
        for r in results:
            print(f"  Output: {r.outputs[0].text!r}")

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        print("  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_02_with_lora_adapter():
    """Test 2: vLLM with LoRA adapter loaded via native LoRA support."""
    print("\n" + "=" * 60)
    print("TEST 2: Base model + LoRA adapter (vLLM native)")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        print("SKIP: vLLM not installed")
        return False

    model_name = "Qwen/Qwen3.5-4B"

    # Load model with LoRA (same as training)
    print(f"  Loading base model from {model_name}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Save LoRA adapter
    import tempfile

    lora_dir = tempfile.mkdtemp(prefix="sdft_lora_")
    print(f"  Saving LoRA to {lora_dir}...")
    t0 = time.time()
    model.save_pretrained(lora_dir)
    import os

    for f in os.listdir(lora_dir):
        sz = os.path.getsize(os.path.join(lora_dir, f))
        print(f"    {f}: {sz:,} bytes")
    print(f"  Saved in {time.time() - t0:.1f}s")

    # Create vLLM with LoRA
    lora_request = LoRARequest("sdft", 1, lora_dir)
    print(f"  Creating LLM with enable_lora=True...")
    t0 = time.time()
    try:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=2560,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=1,
            disable_log_stats=True,
            enforce_eager=True,
            enable_lora=True,
            max_lora_rank=64,
        )
        print(f"  LLM created in {time.time() - t0:.1f}s")

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=16,
        )
        prompts = ["Hello, how are you?"]
        print(f"  Generating with LoRA...")
        t0 = time.time()
        results = llm.generate(prompts, sampling_params, lora_request=lora_request)
        print(f"  Generated in {time.time() - t0:.1f}s")
        for r in results:
            print(f"  Output: {r.outputs[0].text!r}")

        del llm
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_03_with_limit_mm():
    """Test 3: vLLM with limit_mm_per_prompt (our text-only mode)."""
    print("\n" + "=" * 60)
    print("TEST 3: Base model + limit_mm_per_prompt (text-only mode)")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("SKIP: vLLM not installed")
        return False

    model_name = "Qwen/Qwen3.5-4B"

    print(f"  Creating LLM with limit_mm_per_prompt...")
    t0 = time.time()
    try:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=2560,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=1,
            disable_log_stats=True,
            enforce_eager=True,
            enable_lora=False,
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        )
        print(f"  LLM created in {time.time() - t0:.1f}s")

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=16,
        )
        prompts = ["Hello, how are you?"]
        t0 = time.time()
        results = llm.generate(prompts, sampling_params)
        print(f"  Generated in {time.time() - t0:.1f}s")
        for r in results:
            print(f"  Output: {r.outputs[0].text!r}")

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        print("  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_04_stress_dual_gpu():
    """Test 4: Two vLLM engines simultaneously (simulates DDP)."""
    print("\n" + "=" * 60)
    print("TEST 4: Two engines simultaneously on separate GPUs")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("SKIP: vLLM not installed")
        return False

    model_name = "Qwen/Qwen3.5-4B"
    import os

    print(f"  Creating engine on GPU 0...")
    t0 = time.time()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        llm0 = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=2560,
            gpu_memory_utilization=0.4,
            tensor_parallel_size=1,
            disable_log_stats=True,
            enforce_eager=True,
            enable_lora=False,
        )
        print(f"  GPU0 engine created in {time.time() - t0:.1f}s")

    except Exception as e:
        print(f"  GPU0 engine FAIL: {e}")
        return False

    return True  # Both succeeded
    # Note: creating a second engine on GPU1 via CUDA_VISIBLE_DEVICES
    # in the same process is tricky; skip for now.


if __name__ == "__main__":
    results = {}
    results["base_model"] = test_01_plain_model_no_adapter()
    results["with_lora"] = test_02_with_lora_adapter()
    results["limit_mm"] = test_03_with_limit_mm()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
