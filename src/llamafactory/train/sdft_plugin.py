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
#
# sdft_plugin.py
# Single-file Self-Distillation Fine-Tuning (SDFT) plugin for LlamaFactory
# Usage: from sdft_plugin import register_sdft; register_sdft()

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# =============================================================================
# 1. SDFT Configuration (extends LlamaFactory's FinetuningArguments)
# =============================================================================
@dataclass
class SDFTArguments:
    """SDFT-specific hyperparameters - merge these into your training config"""
    stage: str = "sdft"  # Must match stage name in config.yaml

    # Distillation parameters
    alpha: float = field(default=0.0, metadata={"help": "KL type: 0=forward, 1=reverse, 0<x<1=JS"})
    beta: float = field(default=0.0, metadata={"help": "KL coefficient w.r.t. base model for stability"})

    # Loss masking
    num_loss_tokens_to_skip: int = field(default=3, metadata={"help": "Skip first N tokens in loss"})
    top_entropy_quantile: float = field(default=1.0, metadata={"help": "Only loss on top-quantile entropy tokens"})

    # Teacher management
    teacher_model_name: Optional[str] = field(default=None, metadata={"help": "Path to teacher model (None = sync from student)"})
    sync_teacher_every: int = field(default=1, metadata={"help": "Sync teacher weights from student every N steps"})

    # Generation (on-policy sampling)
    num_generations: int = field(default=1, metadata={"help": "Completions to sample per prompt"})
    max_new_tokens: int = field(default=256, metadata={"help": "Max new tokens for on-policy generation"})
    temperature: float = field(default=1.0)
    top_p: float = field(default=1.0)

    # vLLM integration
    use_vllm_for_generation: bool = field(default=True, metadata={"help": "Use vLLM for faster on-policy generation"})
    vllm_sync_every: int = field(default=32, metadata={"help": "Sync training weights to vLLM every N steps (0=never)"})
    vllm_gpu_memory_utilization: float = field(default=0.85, metadata={"help": "GPU memory fraction for vLLM"})

    # Demonstration buffer for teacher-conditioning
    num_demonstrations: int = field(default=2, metadata={"help": "Number of few-shot demos to prepend to teacher prompt"})


# =============================================================================
# 2. Dual-Prompt Data Collator (supports teacher_prompt column)
# =============================================================================
class SDFTDataCollator:
    """Handles batch construction for SDFT.
    Supports two input formats:
    1. Preprocessed (input_ids, labels) — extracts prompt from tokenized data.
    2. Raw text (prompt, teacher_prompt, response) — tokenizes on-the-fly.
    Returns tokenized prompt ids for both student and teacher forward passes.
    """

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _extract_prompt_ids(self, feature: dict) -> torch.Tensor:
        """Extract prompt token ids from a preprocessed feature dict."""
        # Check for raw text columns first (fallback for raw datasets)
        if "prompt" in feature:
            enc = self.tokenizer(
                feature["prompt"],
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return enc["input_ids"][0]

        # Preprocessed format: extract prompt from input_ids using labels mask
        input_ids = torch.tensor(feature["input_ids"])
        labels = feature.get("labels", None)

        if labels is not None:
            labels_t = torch.tensor(labels)
            # Labels switch from IGNORE_INDEX (-100) to real token ids at prompt/response boundary
            non_masked = (labels_t != -100).nonzero(as_tuple=False)
            if non_masked.numel() > 0:
                prompt_len = non_masked[0].item()
            else:
                prompt_len = len(input_ids)  # All prompt, no response
            return input_ids[:prompt_len]

        return input_ids

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {}

        # Extract student prompt ids
        student_ids = [self._extract_prompt_ids(f) for f in features]

        # Pad student prompts
        student_enc = self.tokenizer.pad(
            {"input_ids": student_ids},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["student_input_ids"] = student_enc["input_ids"]
        batch["student_attention_mask"] = student_enc["attention_mask"]

        # Teacher prompts: use raw teacher_prompt if available, else same as student
        has_teacher = any("teacher_prompt" in f for f in features)
        if has_teacher:
            teacher_texts = [f.get("teacher_prompt", "") for f in features]
            teacher_enc = self.tokenizer(
                teacher_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            teacher_enc = student_enc  # Same prompt for both

        batch["teacher_input_ids"] = teacher_enc["input_ids"]
        batch["teacher_attention_mask"] = teacher_enc["attention_mask"]

        # Completion ids (optional, only when raw response column exists)
        if features and "response" in features[0]:
            responses = [f["response"] for f in features]
            response_enc = self.tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["completion_ids"] = response_enc["input_ids"]
            batch["completion_mask"] = response_enc["attention_mask"]

        return batch


# =============================================================================
# 3. SDFT Loss Utilities (KL divergence + entropy filtering)
# =============================================================================
def get_full_distribution_logps(logits: torch.Tensor, shift: bool = True) -> torch.Tensor:
    """Shift logits (predict position i+1) and compute full log-softmax over vocab.

    Args:
        logits: Raw logits [B, L, V]
        shift: If True, drop last logit (no next token to predict from it)

    Returns:
        Full log-softmax distribution [B, L', V] where L' = L-1 if shift else L
    """
    if shift:
        logits = logits[:, :-1, :]
    return F.log_softmax(logits, dim=-1)


def compute_distribution_kl(
    student_full_logps: torch.Tensor,  # [..., V]
    teacher_full_logps: torch.Tensor,  # [..., V]
    alpha: float,
) -> torch.Tensor:
    """Compute KL divergence between full vocabulary distributions.

    Args:
        student_full_logps: Student log-softmax over vocab [..., V]
        teacher_full_logps: Teacher log-softmax over vocab [..., V]
        alpha: 0=forward KL, 1=reverse KL, 0<x<1=Jensen-Shannon

    Returns:
        Per-token KL divergence [*] (summed over vocab dimension)
    """
    if alpha == 0.0:
        kl_raw = F.kl_div(student_full_logps, teacher_full_logps, reduction="none", log_target=True)
    elif alpha == 1.0:
        kl_raw = F.kl_div(teacher_full_logps, student_full_logps, reduction="none", log_target=True)
    else:
        a = torch.tensor(alpha, dtype=student_full_logps.dtype, device=student_full_logps.device)
        log_mix = torch.logaddexp(
            student_full_logps + torch.log(1 - a),
            teacher_full_logps + torch.log(a),
        )
        kl_raw = a * F.kl_div(log_mix, teacher_full_logps, reduction="none", log_target=True) + (
            1 - a
        ) * F.kl_div(log_mix, student_full_logps, reduction="none", log_target=True)
    return kl_raw.sum(dim=-1)  # sum over vocab → per-token KL


def compute_entropy_from_logps(full_logps: torch.Tensor) -> torch.Tensor:
    """Compute entropy H(p) = -sum(p * log(p)) from full log-softmax distribution.

    Args:
        full_logps: Log-softmax over vocab [*, V]

    Returns:
        Per-token entropy [*]
    """
    probs = full_logps.exp()
    return -(probs * full_logps).sum(dim=-1)


def filter_by_entropy(
    entropy: torch.Tensor,
    quantile: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Only keep tokens with highest entropy (most uncertain).

    Args:
        entropy: Per-token entropy values [B, L]
        quantile: Keep top 1-quantile fraction (e.g., 0.8 keeps top 20%)
        mask: Base mask to restrict candidate pool [B, L]

    Returns:
        Float mask of same shape
    """
    if quantile >= 1.0:
        return mask

    masked_entropy = entropy * mask
    flat = masked_entropy[mask.bool()]
    if flat.numel() == 0:
        return mask

    threshold = torch.quantile(flat, 1 - quantile)
    return ((masked_entropy >= threshold) & mask.bool()).float()


# =============================================================================
# 4. SDFT Trainer Wrapper (patches LlamaFactory's SFTTrainer)
# =============================================================================
class SDFTTrainerWrapper:
    """Minimal wrapper that adds SDFT logic to LlamaFactory's existing SFTTrainer.
    Does NOT inherit - instead, we override compute_loss via monkey-patch.
    """

    def __init__(
        self,
        base_trainer,  # LlamaFactory's SFTTrainer instance
        sdft_args: SDFTArguments,
        teacher_model,
        tokenizer,
        data_collator: Optional[SDFTDataCollator] = None,
        vllm_engine=None,
    ):
        self.base = base_trainer
        self.args = sdft_args
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model
        self.data_collator = data_collator or SDFTDataCollator(tokenizer)
        self.vllm_engine = vllm_engine

        # Track steps for teacher sync
        self.global_step = 0

        # Demonstration buffer: stores (prompt_text, response_text) pairs for teacher conditioning
        self._demo_buffer: list[tuple[str, str]] = []
        self._demo_buffer_size = getattr(sdft_args, "num_demonstrations", 2) * 10  # ring buffer

        # Freeze teacher only if it's a separate model (not shared with student)
        if self.teacher_model is not None and self.teacher_model is not base_trainer.model:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        # Patch the base trainer's compute_loss
        self._original_compute_loss = base_trainer.compute_loss
        base_trainer.compute_loss = lambda model, inputs, return_outputs=False, **kwargs: self.compute_loss(
            model, inputs, return_outputs
        )

    def _sync_teacher_weights(self):
        """Copy student weights to teacher (for 2A: same model, synced)"""
        if self.args.teacher_model_name is not None:
            return  # Using external teacher, don't sync
        for t_param, s_param in zip(
            self.teacher_model.parameters(),
            self.base.model.parameters()
        ):
            t_param.data.copy_(s_param.data)

    def _build_teacher_prompt(self, student_prompt: str) -> str:
        """Build a demonstration-conditioned teacher prompt by prepending few-shot examples.

        SDFT requires the teacher to see demonstrations that the student doesn't.
        This produces the distribution difference that KL divergence measures.
        """
        if not self._demo_buffer or self.args.num_demonstrations <= 0:
            return student_prompt

        import random

        num_demos = min(self.args.num_demonstrations, len(self._demo_buffer))
        demos = random.sample(self._demo_buffer, num_demos)
        demo_text = "\n\n".join(f"Example {i+1}:\nQ: {q}\nA: {a}" for i, (q, a) in enumerate(demos))
        return f"{demo_text}\n\nNow answer:\nQ: {student_prompt}\nA:"

    def _update_demo_buffer(self, prompts: list[str], completions: list[str]) -> None:
        """Store prompt-completion pairs for future use as teacher demonstrations."""
        if self._demo_buffer_size <= 0:
            return
        for p, c in zip(prompts, completions):
            # Store a truncated version to keep prompts manageable
            self._demo_buffer.append((p[:512], c[:256]))
        # Trim to ring buffer size
        if len(self._demo_buffer) > self._demo_buffer_size:
            self._demo_buffer = self._demo_buffer[-self._demo_buffer_size:]

    def _generate_completions_on_policy(
        self,
        prompts: list[str],
        use_teacher: bool = False
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Generate completions using on-policy sampling.
        Returns (completions_text, completion_ids) where completion_ids are token ID tensors.
        """
        model = self.teacher_model if use_teacher else self.base.model

        # --- vLLM path (synchronous LLM, configured at training start) ---
        if self.args.use_vllm_for_generation and self.vllm_engine is not None and self.vllm_engine.is_initialized:
            completions_text = self.vllm_engine.generate(prompts)
            if completions_text is not None:
                # Tokenize generated text back to IDs for the forward pass
                completion_ids = [
                    self.tokenizer.encode(c, add_special_tokens=False, return_tensors="pt")[0]
                    for c in completions_text
                ]
                return completions_text, completion_ids

        # --- Fallback: transformers generate ---
        tokenizer_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenizer_inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=True,
                num_return_sequences=self.args.num_generations,
            )

        completions_text = []
        completion_ids = []
        for i in range(len(prompts)):
            prompt_len = len(tokenizer_inputs["input_ids"][i])
            for j in range(self.args.num_generations):
                full_seq = outputs[i * self.args.num_generations + j]
                gen_ids = full_seq[prompt_len:]
                completions_text.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
                completion_ids.append(gen_ids)

        return completions_text, completion_ids

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """SDFT loss: KL(student||teacher) between full vocabulary distributions
        on on-policy generated completions, with demonstration-conditioned teacher.
        """
        self.global_step += 1

        # 1. Decode student prompts and generate completions on-policy
        prompts_text = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in inputs["student_input_ids"]
        ]
        completions_text, completion_ids_list = self._generate_completions_on_policy(
            prompts_text, use_teacher=False
        )

        # 1b. Build demonstration-conditioned teacher prompts
        teacher_prompts_text = [self._build_teacher_prompt(p) for p in prompts_text]

        # 1c. Feed the demo buffer with this batch (will be used in future steps)
        self._update_demo_buffer(prompts_text, completions_text)

        # 2. Pad student prompt IDs (from collator) and completion IDs
        student_prompt_ids_list = [ids for ids in inputs["student_input_ids"]]
        student_prompt_ids = torch.nn.utils.rnn.pad_sequence(
            student_prompt_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        completion_ids = torch.nn.utils.rnn.pad_sequence(
            completion_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        completion_len = completion_ids.size(1)

        # 3. Student forward pass (student prompt + generated completions)
        student_full_ids = torch.cat([student_prompt_ids, completion_ids], dim=1).to(model.device)
        student_full_attn = (student_full_ids != self.tokenizer.pad_token_id).long()

        student_outputs = model(input_ids=student_full_ids, attention_mask=student_full_attn)
        student_logits = student_outputs.logits

        # 4. Teacher forward pass (demo-conditioned teacher prompt + same completions)
        with torch.no_grad():
            teacher_prompt_enc = self.tokenizer(
                teacher_prompts_text,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length - completion_len,
                return_tensors="pt",
            ).to(self.teacher_model.device)

            teacher_prompt_ids = teacher_prompt_enc["input_ids"]
            teacher_prompt_attn = teacher_prompt_enc["attention_mask"]

            teacher_comp_ids = completion_ids.to(self.teacher_model.device)
            teacher_comp_attn = (teacher_comp_ids != self.tokenizer.pad_token_id).long()

            teacher_full_ids = torch.cat([teacher_prompt_ids, teacher_comp_ids], dim=1)
            teacher_full_attn = torch.cat([teacher_prompt_attn, teacher_comp_attn], dim=1)

            teacher_outputs = self.teacher_model(
                input_ids=teacher_full_ids, attention_mask=teacher_full_attn
            )
            teacher_logits = teacher_outputs.logits

        # 5. Trim logits to completion tokens (last completion_len positions, shifted)
        student_logits_comp = student_logits[:, -(completion_len + 1) : -1, :]
        teacher_logits_comp = teacher_logits[:, -(completion_len + 1) : -1, :]

        student_full_logps = get_full_distribution_logps(student_logits_comp, shift=False)
        teacher_full_logps = get_full_distribution_logps(teacher_logits_comp, shift=False)

        # 6. Completion mask (exclude pad tokens)
        completion_mask = (completion_ids != self.tokenizer.pad_token_id).float().to(model.device)

        if self.args.num_loss_tokens_to_skip > 0:
            completion_mask[:, : self.args.num_loss_tokens_to_skip] = 0

        # 7. Entropy-based token filtering
        if self.args.top_entropy_quantile < 1.0:
            student_entropy = compute_entropy_from_logps(student_full_logps)
            completion_mask = completion_mask * filter_by_entropy(
                student_entropy, self.args.top_entropy_quantile, completion_mask
            )

        # 8. KL divergence between full distributions
        per_token_kl = compute_distribution_kl(
            student_full_logps, teacher_full_logps, self.args.alpha
        )  # [B, L_comp]

        # 9. Apply mask and normalize
        masked_kl = (per_token_kl * completion_mask).sum(dim=-1)
        token_counts = completion_mask.sum(dim=-1).clamp(min=1.0)
        kl_loss = (masked_kl / token_counts).mean()

        # 10. Optional beta-KL to frozen reference model
        total_loss = kl_loss
        if self.args.beta > 0.0 and hasattr(self.base, "ref_model") and self.base.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.base.ref_model(
                    input_ids=student_full_ids, attention_mask=student_full_attn
                )
                ref_logits_comp = ref_outputs.logits[:, -(completion_len + 1) : -1, :]
                ref_full_logps = get_full_distribution_logps(ref_logits_comp, shift=False)
            ref_kl = compute_distribution_kl(student_full_logps, ref_full_logps, alpha=0.0)
            ref_masked_kl = (ref_kl * completion_mask).sum(dim=-1) / token_counts
            total_loss = total_loss + self.args.beta * ref_masked_kl.mean()

        # 11. Sync teacher weights if shared
        if self.args.teacher_model_name is None and self.global_step % self.args.sync_teacher_every == 0:
            self._sync_teacher_weights()

        if return_outputs:
            return total_loss, {"kl_loss": kl_loss.detach(), "total_loss": total_loss.detach()}
        return total_loss

    def restore(self):
        """Unpatch the base trainer (cleanup)"""
        self.base.compute_loss = self._original_compute_loss


# =============================================================================
# 5. Registration Hook (call this to activate SDFT in LlamaFactory)
# =============================================================================
def register_sdft(
    trainer_class=None,
    config_class=None,
    enable_logging: bool = True
):
    """Register SDFT plugin with LlamaFactory.
    
    Usage in your training script:
    ```
    from sdft_plugin import register_sdft, SDFTArguments
    register_sdft()
    
    # Then in your config.yaml:
    # stage: sdft
    # alpha: 0.0
    # teacher_model_name: null
    # ...
    ```
    """
    if enable_logging:
        logger.info("🔌 SDFT plugin registered. Use stage='sdft' in config to enable.")

    # Optional: Auto-patch LlamaFactory's trainer factory if classes provided
    if trainer_class and config_class:
        # This is where you'd inject SDFTArguments into FinetuningArguments
        # and add stage handler to get_trainer() - left as exercise for repo integration
        logger.warning("Auto-patching not implemented in single-file mode. "
                      "See LlamaFactory docs for extending trainer factory.")

    return {
        "SDFTArguments": SDFTArguments,
        "SDFTDataCollator": SDFTDataCollator,
        "SDFTTrainerWrapper": SDFTTrainerWrapper,
        "compute_kl_divergence": compute_kl_divergence
    }


# =============================================================================
# 6. Quick Test / Demo (run: python sdft_plugin.py)
# =============================================================================
if __name__ == "__main__":
    print("✅ SDFT Plugin loaded successfully")
    print(f"✅ Available components: {list(register_sdft(enable_logging=False).keys())}")
    print("\n📋 Next steps:")
    print("1. Add SDFTArguments fields to your LlamaFactory config")
    print("2. Use data collator: SDFTDataCollator(tokenizer)")
    print("3. Wrap trainer: SDFTTrainerWrapper(base_trainer, args, teacher_model, tokenizer)")
    print("4. Set stage: sdft in config.yaml")
    print("\n🔗 See examples/train_lora/ for YAML template")
