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
import json
import os
import signal
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

    **Model compatibility**:
        Downloads auxiliary config files (preprocessor, tokenizer, etc.) from the
        original HF model repo so vLLM's loader finds everything it expects. Works
        with any model architecture — text-only, vision-language, audio, etc.
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
        """Initialize vLLM engine from the current (possibly LoRA-adapted) model weights.

        Args:
            model: The training model (may be a PeftModel with LoRA adapters).
            tokenizer: Tokenizer matching the model.
            model_name_or_path: Original HF model ID or path used to fetch
                auxiliary config files (preprocessor, tokenizer, etc.).
            cutoff_len: Maximum prompt token length.
            max_new_tokens: Maximum tokens to generate per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            gpu_memory_utilization: Fraction of GPU memory for vLLM (0.0-1.0).
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            disable_multimodal: If True, force text-only mode even for multi-modal
                architectures (Qwen3, etc.). Set False for vision/audio datasets.
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

        # Build a complete model directory that vLLM can load
        self._temp_dir = tempfile.mkdtemp(prefix="sdft_vllm_")

        # Phase 1: pull auxiliary config files from original HF repo
        _download_auxiliary_configs(model_name_or_path, self._temp_dir, disable_multimodal=disable_multimodal)

        # Phase 2: save merged weights + model config
        try:
            self._save_merged_weights(model, self._temp_dir)
        except Exception as e:
            print(f"SDFTVLLMEngine: Failed to save merged model weights: {e}")
            return

        # Phase 2b: if text-only mode, strip multi-modal configs that trigger vLLM
        # processor loading (image/video processor deprecation warnings + extra memory)
        if disable_multimodal:
            _strip_multimodal_config(self._temp_dir)

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
                enforce_eager=True,  # skip CUDA graph capture for faster init
            )

            self._sampling_params = self._SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                skip_special_tokens=True,
            )
            self._initialized = True

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

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM: clean up vLLM workers then re-raise."""
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
            # Kill vLLM worker processes spawned during this session
            subprocess.run(
                ["pkill", "-f", "vllm.*EngineCore"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    @staticmethod
    def _save_merged_weights(model, save_dir: str) -> None:
        """Merge LoRA adapters into base weights and save full model to disk.

        Uses ``merge_adapter()`` / ``unmerge_adapter()`` to temporarily merge
        LoRA without destroying the PeftModel wrapper, preserving the training
        state for continued fine-tuning.

        The merged weights are saved as ``pytorch_model.bin`` alongside the
        model config (``config.json``). Auxiliary config files should already
        exist in ``save_dir`` from ``_download_auxiliary_configs``.
        """
        has_lora = hasattr(model, "peft_config") and model.peft_config

        if has_lora:
            # Temporarily merge LoRA into base weights
            model.merge_adapter()

            # Access the underlying transformers model
            if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                underlying = model.base_model.model
            else:
                underlying = model

            # Save merged weights
            torch.save(underlying.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

            # Save model config (overwrites any config from auxiliary download)
            if hasattr(underlying, "config"):
                underlying.config.save_pretrained(save_dir)

            # Restore LoRA training state
            model.unmerge_adapter()
        else:
            # No adapter: full fine-tune or frozen — save directly
            torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            if hasattr(model, "config"):
                model.config.save_pretrained(save_dir)


def _download_auxiliary_configs(model_name_or_path: str, save_dir: str, disable_multimodal: bool = True) -> None:
    """Download non-weight config files from the original HF model repo.

    vLLM's model loader expects a complete model directory with all auxiliary
    config files (preprocessor_config.json, tokenizer_config.json,
    chat_template.jinja, etc.). This downloads those files from the original
    source so vLLM finds everything it needs regardless of architecture.

    When ``disable_multimodal=True`` (default), skips the HF download entirely
    and writes a text-only stub preprocessor config. This forces vLLM to treat
    even multi-modal models as text-only, avoiding image/video processor
    loading errors when the dataset doesn't need multi-modal outputs.

    Args:
        model_name_or_path: HF model ID or local path.
        save_dir: Directory to save config files into.
        disable_multimodal: If True, force text-only mode (skip multi-modal
            processor loading).
    """
    # Route 1: Text-only forced mode — write stub, skip HF download
    if disable_multimodal:
        _write_stub_preprocessor_config(save_dir)
        return

    # Route 2: Full multi-modal — download real configs from HF hub
    is_hf_hub = not os.path.isdir(model_name_or_path) and "/" in model_name_or_path
    if not is_hf_hub:
        _write_stub_preprocessor_config(save_dir)
        return

    try:
        from huggingface_hub import list_repo_files, snapshot_download

        # List all files in the repo
        repo_files = list_repo_files(model_name_or_path)

        # Filter to config-only files (skip weight files)
        weight_extensions = (".safetensors", ".bin", ".pt", ".h5", ".msgpack", ".ot", ".ckpt")
        config_files = [
            f for f in repo_files
            if not any(f.endswith(ext) for ext in weight_extensions)
            and not f.startswith(".")
        ]

        # Download config files to the temp directory
        snapshot_download(
            repo_id=model_name_or_path,
            local_dir=save_dir,
            allow_patterns=config_files,
            local_dir_use_symlinks=False,
        )
    except Exception:
        # Network unavailable or invalid repo — write a stub that works
        # for text-only models; multi-modal models will fail gracefully
        _write_stub_preprocessor_config(save_dir)


def _write_stub_preprocessor_config(save_dir: str) -> None:
    """Write a stub preprocessor config for text-only models.

    Some architectures (e.g., Qwen3) trigger vLLM's multi-modal processor
    loader even for text-only variants. This stub tells vLLM there is no
    image/video processor, allowing text-only generation to proceed.
    Multi-modal models will get their real preprocessor config from the
    HF hub download path above.
    """
    preprocessor_path = os.path.join(save_dir, "preprocessor_config.json")
    if not os.path.exists(preprocessor_path):
        with open(preprocessor_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_processor_type": None,
                    "feature_extractor_type": None,
                    "processor_class": "AutoProcessor",
                },
                f,
            )


def _strip_multimodal_config(save_dir: str) -> None:
    """Remove multi-modal fields from saved config files.

    vLLM v0.20 inspects config.json for processor hints even when
    preprocessor_config.json says text-only. This removes vision/audio
    processor references from the saved config so vLLM doesn't try to
    load unnecessary processors.
    """
    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Remove fields that trigger multi-modal processor loading
    mm_fields = [
        "image_processor_type",
        "video_processor_type",
        "audio_processor_type",
        "processor_class",
        "vision_config",
        "mm_hidden_size",
        "mm_vision_tower",
        "mm_audio_tower",
        "mm_projector_type",
        "image_token_id",
        "video_token_id",
        "audio_token_id",
    ]
    for field in mm_fields:
        config.pop(field, None)

    # Also remove deprecated image_processor_type from preprocessor
    preprocessor_path = os.path.join(save_dir, "preprocessor_config.json")
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, encoding="utf-8") as f:
            pp_config = json.load(f)
        pp_config["image_processor_type"] = None
        pp_config["feature_extractor_type"] = None
        with open(preprocessor_path, "w", encoding="utf-8") as f:
            json.dump(pp_config, f, indent=2)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
