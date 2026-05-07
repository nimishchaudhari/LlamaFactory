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

from typing import TYPE_CHECKING, Optional

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ...train.sdft_plugin import SDFTArguments, SDFTDataCollator, SDFTTrainerWrapper
from ..sft.trainer import CustomSeq2SeqTrainer
from ..trainer_utils import create_modelcard_and_push


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sdft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # Setup teacher model
    teacher_model_name = finetuning_args.teacher_model_name
    if teacher_model_name is None or teacher_model_name == "":
        teacher_model = model
    else:
        from transformers import AutoModelForCausalLM

        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=model_args.torch_dtype,
            device_map=model_args.device_map,
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    # Build SDFT args dataclass
    sdft_args = SDFTArguments(
        stage="sdft",
        alpha=finetuning_args.alpha,
        beta=finetuning_args.beta,
        num_loss_tokens_to_skip=finetuning_args.num_loss_tokens_to_skip,
        top_entropy_quantile=finetuning_args.top_entropy_quantile,
        teacher_model_name=teacher_model_name,
        sync_teacher_every=finetuning_args.sync_teacher_every,
        num_generations=finetuning_args.num_generations,
        max_new_tokens=generating_args.max_new_tokens,
        temperature=generating_args.temperature,
        top_p=generating_args.top_p,
        use_vllm_for_generation=finetuning_args.use_vllm_for_generation,
    )

    # SDFT data collator (produces student/teacher input batches)
    data_collator = SDFTDataCollator(tokenizer, max_length=data_args.cutoff_len)

    # Preserve raw dataset columns (prompt, teacher_prompt, response) for collator
    training_args.remove_unused_columns = False

    # Create base trainer (patched by SDFT wrapper)
    base_trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Wrap with SDFT logic (monkey-patches compute_loss on base_trainer)
    sdft_trainer = SDFTTrainerWrapper(
        base_trainer=base_trainer,
        sdft_args=sdft_args,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training (runs through base_trainer with SDFT-patched compute_loss)
    if training_args.do_train:
        train_result = base_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        base_trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sdft"
            )

        base_trainer.log_metrics("train", train_result.metrics)
        base_trainer.save_metrics("train", train_result.metrics)
        base_trainer.save_state()
        if base_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = base_trainer.evaluate(metric_key_prefix="eval")
        base_trainer.log_metrics("eval", metrics)
        base_trainer.save_metrics("eval", metrics)

    # Cleanup: un-patch base trainer
    sdft_trainer.restore()

    # Create model card
    create_modelcard_and_push(base_trainer, model_args, data_args, training_args, finetuning_args)
