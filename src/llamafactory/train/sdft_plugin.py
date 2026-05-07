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

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

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
    use_vllm_for_generation: bool = field(default=True, metadata={"help": "Use LlamaFactory's vLLM for faster generation"})


# =============================================================================
# 2. Dual-Prompt Data Collator (supports teacher_prompt column)
# =============================================================================
class SDFTDataCollator:
    """
    Handles batch construction for SDFT.
    Supports two input formats:
    1. Preprocessed (input_ids, labels) — extracts prompt from tokenized data.
    2. Raw text (prompt, teacher_prompt, response) — tokenizes on-the-fly.
    Returns tokenized prompt ids for both student and teacher forward passes.
    """

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _extract_prompt_ids(self, feature: Dict) -> torch.Tensor:
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

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
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
def get_batch_logps(
    logits: torch.FloatTensor, 
    labels: torch.LongTensor, 
    average_log_prob: bool = False
) -> torch.FloatTensor:
    """Compute log-probs per token, same as DPO/SDFT implementations"""
    if logits.shape[:-1] != labels.shape:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
    
    loss_mask = (labels != -100)
    labels = labels.masked_fill(~loss_mask, 0)
    
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    return per_token_logps


def compute_kl_divergence(
    student_logps: torch.Tensor,
    teacher_logps: torch.Tensor,
    alpha: float,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence with alpha parameter:
    - alpha=0: forward KL (student||teacher)
    - alpha=1: reverse KL (teacher||student)  
    - 0<alpha<1: Jensen-Shannon style
    """
    if alpha == 0.0:
        # Forward KL: E_teacher[log(teacher/student)]
        kl = F.kl_div(student_logps, teacher_logps, reduction='none', log_target=True)
    elif alpha == 1.0:
        # Reverse KL: E_student[log(student/teacher)]
        kl = F.kl_div(teacher_logps, student_logps, reduction='none', log_target=True)
    else:
        # Jensen-Shannon: mixture distribution
        log_mix = torch.logaddexp(
            student_logps + torch.log(torch.tensor(alpha)),
            teacher_logps + torch.log(torch.tensor(1 - alpha))
        )
        kl = (alpha * F.kl_div(student_logps, log_mix, reduction='none', log_target=True) +
              (1 - alpha) * F.kl_div(teacher_logps, log_mix, reduction='none', log_target=True))
    
    # Apply mask and reduce
    masked_kl = kl * mask
    return masked_kl.sum(-1) / mask.sum(-1).clamp(min=1.0)


def filter_by_entropy(
    logps: torch.Tensor, 
    quantile: float, 
    mask: torch.Tensor
) -> torch.Tensor:
    """Only compute loss on tokens with highest entropy (most uncertain)"""
    if quantile >= 1.0:
        return mask
    
    # Compute entropy per token (approx via log-prob variance)
    entropy = -torch.exp(logps) * logps  # Simplified entropy estimate
    entropy = entropy * mask
    
    # Find threshold for top-quantile
    flat_entropy = entropy[mask.bool()].flatten()
    if flat_entropy.numel() == 0:
        return mask
        
    threshold = torch.quantile(flat_entropy, 1 - quantile)
    high_entropy_mask = (entropy >= threshold) & mask.bool()
    
    return high_entropy_mask.float()


# =============================================================================
# 4. SDFT Trainer Wrapper (patches LlamaFactory's SFTTrainer)
# =============================================================================
class SDFTTrainerWrapper:
    """
    Minimal wrapper that adds SDFT logic to LlamaFactory's existing SFTTrainer.
    Does NOT inherit - instead, we override compute_loss via monkey-patch.
    """
    
    def __init__(
        self,
        base_trainer,  # LlamaFactory's SFTTrainer instance
        sdft_args: SDFTArguments,
        teacher_model,
        tokenizer,
        data_collator: Optional[SDFTDataCollator] = None
    ):
        self.base = base_trainer
        self.args = sdft_args
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model
        self.data_collator = data_collator or SDFTDataCollator(tokenizer)
        
        # Ensure teacher is in eval mode, no gradients
        if self.teacher_model:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
                
        # Track steps for teacher sync
        self.global_step = 0
        
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
            
    def _generate_completions_on_policy(
        self, 
        prompts: List[str], 
        use_teacher: bool = False
    ) -> List[List[int]]:
        """
        Generate completions using on-policy sampling.
        Leverages LlamaFactory's vLLM integration if available (4A).
        """
        model = self.teacher_model if use_teacher else self.base.model
        
        # Check if vLLM is available in LlamaFactory
        try:
            from llamafactory.chat.vllm_engine import VLLMEngine
            if self.args.use_vllm_for_generation and hasattr(self.base, 'vllm_engine'):
                # Use existing vLLM engine from LlamaFactory
                completions = self.base.vllm_engine.generate(
                    prompts,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    n=self.args.num_generations
                )
                return completions
        except ImportError:
            pass
            
        # Fallback: transformers generate
        inputs = self.tokenizer(
            prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=True,
                num_return_sequences=self.args.num_generations
            )
        
        # Decode and return
        completions = []
        for i, prompt in enumerate(prompts):
            prompt_len = len(inputs["input_ids"][i])
            for j in range(self.args.num_generations):
                gen_ids = outputs[i * self.args.num_generations + j][prompt_len:]
                completions.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
                
        return completions
        
    def compute_loss(
        self, 
        model, 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        SDFT loss: KL(student||teacher) on on-policy generated completions
        """
        self.global_step += 1
        
        # 1. Generate completions on-policy (from student)
        prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                  for ids in inputs["student_input_ids"]]
        completions = self._generate_completions_on_policy(prompts, use_teacher=False)
        
        # 2. Build full sequences for forward pass
        student_seqs = [p + c for p, c in zip(prompts, completions)]
        teacher_seqs = [f.get("teacher_prompt", p) + c for p, c, f in zip(prompts, completions, [inputs]*len(prompts))]
        
        # 3. Forward pass: student model
        student_enc = self.tokenizer(
            student_seqs, padding=True, return_tensors="pt"
        ).to(model.device)
        student_outputs = model(**student_enc)
        student_logps = get_batch_logps(
            student_outputs.logits, 
            student_enc["input_ids"], 
            average_log_prob=False
        )
        
        # 4. Forward pass: teacher model (no grad)
        with torch.no_grad():
            teacher_enc = self.tokenizer(
                teacher_seqs, padding=True, return_tensors="pt"
            ).to(self.teacher_model.device)
            teacher_outputs = self.teacher_model(**teacher_enc)
            teacher_logps = get_batch_logps(
                teacher_outputs.logits,
                teacher_enc["input_ids"],
                average_log_prob=False
            )
        
        # 5. Compute loss mask (skip first N tokens, apply response mask)
        labels = student_enc["input_ids"].clone()
        loss_mask = (labels != self.tokenizer.pad_token_id).float()
        
        # Skip initial tokens (e.g., prompt + special tokens)
        if self.args.num_loss_tokens_to_skip > 0:
            loss_mask[:, :self.args.num_loss_tokens_to_skip] = 0
            
        # Only compute loss on completion tokens (not prompt)
        # This assumes response starts after first eos_token
        eos_positions = (labels == self.tokenizer.eos_token_id).float().argmax(dim=1)
        for i, pos in enumerate(eos_positions):
            if pos > 0:
                loss_mask[i, :pos+1] = 0  # Mask prompt + eos
        
        # 6. Entropy-based token filtering
        if self.args.top_entropy_quantile < 1.0:
            loss_mask = loss_mask * filter_by_entropy(
                student_logps, self.args.top_entropy_quantile, loss_mask
            )
        
        # 7. Compute KL divergence loss
        kl_loss = compute_kl_divergence(
            student_logps, teacher_logps, self.args.alpha, loss_mask
        )
        
        # 8. Optional: KL regularization w.r.t. base model (beta term)
        total_loss = kl_loss
        if self.args.beta > 0.0 and hasattr(self.base, 'ref_model') and self.base.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.base.ref_model(**student_enc)
                ref_logps = get_batch_logps(ref_outputs.logits, labels, average_log_prob=False)
            reg_kl = compute_kl_divergence(student_logps, ref_logps, alpha=0.0, mask=loss_mask)
            total_loss = total_loss + self.args.beta * reg_kl
        
        # 9. Sync teacher weights if needed
        if self.args.teacher_model_name is None and self.global_step % self.args.sync_teacher_every == 0:
            self._sync_teacher_weights()
        
        loss = total_loss.mean()
        
        if return_outputs:
            return loss, {"kl_loss": kl_loss.detach(), "total_loss": loss.detach()}
        return loss
        
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
    """
    Register SDFT plugin with LlamaFactory.
    
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