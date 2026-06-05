
"""
Inherits from llamafactory.train.sft.workflow.
Run Safety-based Fine-tuning method such as SafeGrad.
"""
import copy
from typing import TYPE_CHECKING, Optional, Literal
from llamafactory.data import SFTDataCollatorWith4DAttentionMask
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor

from saft.src.abs import ABSTrainer
from saft.src._data_nopoison import get_saft_dataset, get_saft_template_and_fix_tokenizer

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llamafactory.hparams import FinetuningArguments, GeneratingArguments
    from saft.src._args import SaFTDataArguments, SaFTTrainingArguments, SaFTModelArguments

logger = get_logger(__name__)


def run_saft_abs(
    model_args: "SaFTModelArguments",
    data_args: "SaFTDataArguments",
    training_args: "SaFTTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
    abs_revert: bool = False,
    abs_projection: bool = True,
    abs_loss_type: Literal["KL", "CE"] = "CE",
    abs_rescale: bool = False,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    template = get_saft_template_and_fix_tokenizer(
        tokenizer,
        data_args.template,
        data_args
    )
    aligned_template = get_saft_template_and_fix_tokenizer(
        tokenizer,
        data_args.aligned_template,
        data_args
    )
    unaligned_template = get_saft_template_and_fix_tokenizer(
        tokenizer,
        data_args.unaligned_template,
        data_args
    )

    dataset_module = get_saft_dataset(
        template=template,
        aligned_template=aligned_template,
        unaligned_template=unaligned_template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    aligned_model_args = copy.deepcopy(model_args)
    aligned_model_args.model_name_or_path = aligned_model_args.aligned_model_name_or_path
    aligned_model = load_model(tokenizer, aligned_model_args, finetuning_args, False)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    if model_args.use_kt:
        raise NotImplementedError("`use_kt` is not supported in KTransformers SFT yet.")
        # from ktransformers.sft.lora import KTrainer  # type: ignore
        # from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore
        #
        # GLOBAL_CONFIG._config["mod"] = "sft"
        #
        # trainer = KTrainer(
        #     model=model,
        #     args=training_args,
        #     tokenizer=tokenizer_module,
        #     data_collator=data_collator,
        #     callbacks=callbacks,
        #     **dataset_module,
        #     **metric_module,
        # )
        # trainer.model_accepts_loss_kwargs = False
        # model.config.use_cache = False

    else:
        trainer = ABSTrainer(
            model=model,
            aligned_model=aligned_model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
            abs_revert=abs_revert,
            abs_projection=abs_projection,
            abs_loss_type=abs_loss_type,
            abs_rescale=abs_rescale
        )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)


# if __name__ == '__main__':
