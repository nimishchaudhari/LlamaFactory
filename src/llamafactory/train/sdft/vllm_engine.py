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

import gc
import os
import tempfile
from typing import TYPE_CHECKING, Optional

import torch


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class SDFTVLLMEngine:
    """Wraps vLLM's synchronous LLM for fast on-policy generation during SDFT training.

    Replaces the slow ``model.generate()`` fallback with vLLM's PagedAttention-based
    batch generation (10-50x faster for long completions).

    **Weight sync strategy (Phase 1)**:
        The engine is created once at training start from the initial merged LoRA
        weights. Generation is slightly off-policy as training progresses, but the
        demonstration buffer provides the primary KL signal. Full periodic weight
        sync is planned for Phase 2.

    Usage::

        engine = SDFTVLLMEngine(model, tokenizer, model_name_or_path, cutoff_len, max_new_tokens)
        completions = engine.generate(prompts)  # or None if fallback needed
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
    ):
        """Initialize vLLM engine from the current (possibly LoRA-adapted) model weights.

        Args:
            model: The training model (may be a PeftModel with LoRA adapters).
            tokenizer: Tokenizer matching the model.
            model_name_or_path: Original HF model ID or path (unused; compatibility).
            cutoff_len: Maximum prompt token length.
            max_new_tokens: Maximum tokens to generate per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            gpu_memory_utilization: Fraction of GPU memory for vLLM (0.0-1.0).
            tensor_parallel_size: Number of GPUs for tensor parallelism.
        """
        self._llm = None
        self._initialized = False
        self._temp_dir: Optional[str] = None

        # Check vLLM availability
        try:
            from vllm import LLM, SamplingParams

            self._LLM = LLM
            self._SamplingParams = SamplingParams
        except ImportError:
            print("SDFTVLLMEngine: vLLM not installed, falling back to model.generate().")
            return

        # Save merged weights to temporary dir for vLLM loading
        self._temp_dir = tempfile.mkdtemp(prefix="sdft_vllm_")

        try:
            self._save_merged_weights(model, self._temp_dir)
        except Exception as e:
            print(f"SDFTVLLMEngine: Failed to save merged model weights: {e}")
            return

        # Build and initialize vLLM engine
        try:
            self._llm = self._LLM(
                model=self._temp_dir,
                trust_remote_code=True,
                dtype="auto",
                max_model_len=cutoff_len + max_new_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                disable_log_stats=True,
                enable_lora=False,
            )

            self._sampling_params = self._SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                skip_special_tokens=True,
            )
            self._initialized = True
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
            List of completion strings, or None if engine is not initialized
            (caller should fall back to ``model.generate()``).
        """
        if not self.is_initialized:
            return None

        results = self._llm.generate(prompts, self._sampling_params)
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

        if self._temp_dir and os.path.isdir(self._temp_dir):
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    @staticmethod
    def _save_merged_weights(model, save_dir: str) -> None:
        """Merge LoRA adapters into base weights and save full model to disk.

        Uses ``merge_adapter()`` / ``unmerge_adapter()`` to temporarily merge
        LoRA without destroying the PeftModel wrapper, preserving the training
        state for continued fine-tuning.
        """
        HAS_CONFIG = hasattr(model, "config")
        has_lora = hasattr(model, "peft_config") and model.peft_config

        if has_lora:
            # Temporarily merge LoRA into base weights
            model.merge_adapter()

            # Access the underlying transformers model
            if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                underlying = model.base_model.model
            else:
                underlying = model

            # Save full model weights
            torch.save(underlying.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

            # Save model config for vLLM compatibility
            if hasattr(underlying, "config"):
                underlying.config.save_pretrained(save_dir)

            # Restore LoRA training state
            model.unmerge_adapter()
        else:
            # No adapter: full fine-tune or frozen — save directly
            torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            if HAS_CONFIG:
                model.config.save_pretrained(save_dir)
