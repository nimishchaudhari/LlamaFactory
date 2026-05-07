# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import gc
import os
import signal
import tempfile
from typing import TYPE_CHECKING, Optional

import torch


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class SDFTVLLMEngine:
    """Wraps vLLM's synchronous LLM for fast on-policy generation during SDFT training.

    Instead of merging LoRA weights and saving a full model copy (~9GB), we save
    only the LoRA adapter (<50MB) and point vLLM at the original model path with
    ``enable_lora=True`` + ``LoRARequest``. vLLM applies the adapter on-the-fly,
    eliminating disk I/O, serialization issues, and ``/tmp`` quota problems.

    Usage::

        engine = SDFTVLLMEngine(model, tokenizer, model_name_or_path, cutoff_len, max_new_tokens)
        completions = engine.generate(prompts)
        engine.shutdown()
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        model_name_or_path: str,
        cutoff_len: int = 2048,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        disable_multimodal: bool = True,
    ):
        """Initialize vLLM engine using native LoRA support.

        Args:
            model: The training PeftModel with LoRA adapters.
            tokenizer: Tokenizer matching the model.
            model_name_or_path: Original HF model ID or path (loaded by vLLM from cache).
            cutoff_len: Maximum prompt token length.
            max_new_tokens: Maximum tokens to generate per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            gpu_memory_utilization: Fraction of GPU memory for vLLM (0.0-1.0).
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            disable_multimodal: If True, force text-only mode.
        """
        self._llm = None
        self._initialized = False
        self._lora_dir: Optional[str] = None

        # Check vLLM availability
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest

            self._LLM = LLM
            self._SamplingParams = SamplingParams
            self._LoRARequest = LoRARequest
        except ImportError:
            print("SDFTVLLMEngine: vLLM not installed, falling back to model.generate().")
            return

        # Determine if we have a LoRA model
        has_lora = hasattr(model, "peft_config") and model.peft_config is not None
        base_model_path = model_name_or_path

        # If LoRA: save adapter to temp dir, load base model in vLLM with LoRA
        if has_lora and hasattr(model, "save_pretrained"):
            self._lora_dir = tempfile.mkdtemp(prefix="sdft_lora_")
            try:
                model.save_pretrained(self._lora_dir)
                self._lora_request = self._LoRARequest("sdft", 1, self._lora_dir)
            except Exception as e:
                print(f"SDFTVLLMEngine: Failed to save LoRA adapter: {e}")
                self._lora_dir = None
                self._lora_request = None
        else:
            self._lora_request = None

        # Build and initialize vLLM engine pointing at the ORIGINAL model
        try:
            print(f"SDFTVLLMEngine: Initializing vLLM from {base_model_path}...")

            engine_kwargs = {
                "model": base_model_path,
                "trust_remote_code": True,
                "dtype": "auto",
                "max_model_len": cutoff_len + max_new_tokens,
                "gpu_memory_utilization": gpu_memory_utilization,
                "tensor_parallel_size": tensor_parallel_size,
                "disable_log_stats": True,
                "enable_lora": self._lora_request is not None,
                "max_lora_rank": 64,
                "enforce_eager": True,  # skip CUDA graph capture for fast startup
            }

            # In text-only mode, skip multi-modal processor loading
            if disable_multimodal:
                engine_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}

            self._llm = self._LLM(**engine_kwargs)

            self._sampling_params = self._SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                skip_special_tokens=True,
            )
            self._initialized = True
            print("SDFTVLLMEngine: vLLM initialized successfully.")

            # Register cleanup on process exit or interrupt (Ctrl+C)
            atexit.register(self._cleanup_workers)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception as e:
            print(f"SDFTVLLMEngine: Failed to initialize vLLM: {e}")
            self._llm = None

    @property
    def is_initialized(self) -> bool:
        """Check if the vLLM engine is ready for generation."""
        return self._initialized and self._llm is not None

    def generate(self, prompts: list[str]) -> Optional[list[str]]:
        """Generate completions for a batch of text prompts.

        Args:
            prompts: List of prompt strings to complete.

        Returns:
            List of completion strings, or None if engine is not initialized.
        """
        if not self.is_initialized:
            return None

        results = self._llm.generate(
            prompts, self._sampling_params, lora_request=self._lora_request
        )
        return [result.outputs[0].text for result in results]

    def shutdown(self) -> None:
        """Release vLLM engine and clean up temporary files."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._initialized = False

        if self._lora_dir and os.path.isdir(self._lora_dir):
            import shutil

            shutil.rmtree(self._lora_dir, ignore_errors=True)
            self._lora_dir = None

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM: clean up then re-raise."""
        self.shutdown()
        self._force_kill_workers()
        raise KeyboardInterrupt

    @staticmethod
    def _cleanup_workers():
        """Atexit handler: kill orphaned vLLM worker processes."""
        SDFTVLLMEngine._force_kill_workers()

    @staticmethod
    def _force_kill_workers():
        """Force-kill any lingering vLLM EngineCore processes."""
        import subprocess

        try:
            subprocess.run(
                ["pkill", "-f", "vllm.*EngineCore"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
