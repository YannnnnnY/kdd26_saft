from pathlib import Path
from typing import Optional, Union, Literal, Any
from dataclasses import dataclass, field

from transformers.utils import add_start_docstrings
from transformers.generation.configuration_utils import GenerationConfig

from llamafactory.extras import logging
from llamafactory.hparams import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
    parser
)

logger = logging.get_logger(__name__)


def get_saft_train_args(args: dict[str, Any] | list[str] | None = None) -> parser._TRAIN_CLS:
    parser._TRAIN_ARGS = [
        SaFTModelArguments,
        SaFTDataArguments,
        SaFTTrainingArguments,
        FinetuningArguments,
        GeneratingArguments
    ]
    parser._TRAIN_CLS = tuple[
        SaFTModelArguments,
        SaFTDataArguments,
        SaFTTrainingArguments,
        FinetuningArguments,
        GeneratingArguments
    ]
    return parser.get_train_args(args)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class SaFTTrainingArguments(TrainingArguments):
    """
    Safety-based Fine-tuning training arguments.
    Inherits from `transformers.Seq2SeqTrainingArguments`.
    """
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    saft_trainer: Optional[str] = field(
        default=Literal["safegrad", "asymgrad"],
        metadata={"help": "Specify the safety fine-tune trainer to use."},
    )
    align_gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    per_gpu_align_batch_size: int = field(
        default=None, metadata={"help": "Batch size per device accelerator core/CPU for training."}
    )
    per_device_align_batch_size: int = field(
        default=None, metadata={"help": "Batch size per device accelerator core/CPU for training."}
    )
    rho: float = field(
        default=1.0, metadata={"help": "Safety gradient trade-off parameter."}
    )
    projection: bool = field(
        default=True, metadata={"help": "Whether to use projection to compute safety gradients."}
    )
    abs_projection: bool = field(
        default=True, metadata={"help": "For ablation only: Whether to use projection to compute safety gradients."}
    )
    abs_revert: bool = field(
        default=False, metadata={"help": "For ablation only: Whether to project safety gradient to task gradient."}
    )
    abs_loss_type: str = field(
        default="CE", metadata={"help": "For ablation only: Loss type for safety gradients."}
    )
    abs_rescale: bool = field(
        default=False, metadata={"help": "For ablation only: Rescale task grad after projection."}
    )

    def __post_init__(self):
        super().__post_init__()

    @property
    def align_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_align_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_align_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_align_batch_size` is preferred."
            )
        per_device_align_batch_size = self.per_device_align_batch_size or self.per_gpu_align_batch_size
        align_batch_size = per_device_align_batch_size * max(1, self.n_gpu)
        return align_batch_size



@dataclass
@add_start_docstrings(ModelArguments.__doc__)
class SaFTModelArguments(ModelArguments):
    """
    """
    aligned_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
@add_start_docstrings(DataArguments.__doc__)
class SaFTDataArguments(DataArguments):
    """
    Safety-based Fine-tuning dataset arguments.
    Inherits from `llamafactory.hparams.DataArguments`.
    - unaligned: dataset to be mixed into benign dataset.
    - aligned: dataset to be used for safety gradient.
    Note: Don't set max_samples, set benign_num instead.
    """
    benign_num: int = field(
        default=None,
        metadata={"help": "Discrete samples of benign dataset to be used."},
    )

    unaligned_template: str | None = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )

    unaligned_dataset: str | None = field(
        default=None,
        metadata={"help": "The name of unaligned dataset(s). To be mixed into benign train dataset."},
    )

    unaligned_ratio: float = field(
        default=0.1,
        metadata={"help": "The ratio of unaligned samples to be mixed."},
    )

    unaligned_num: int = field(
        default=None,
        metadata={"help": "Discrete samples of unaligned datasets to be mixed."},
    )

    aligned_template: str | None = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )

    aligned_stu_template: str | None = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )

    aligned_dataset: str | None = field(
        default=None,
        metadata={"help": "The name of aligned dataset(s) to use for get safety gradients. Use commas to separate multiple datasets."},
    )

    aligned_ratio: float = field(
        default=1.0,
        metadata={"help": "The ratio of aligned samples to be used compare to all the train samples (benign + unaligned)."},
    )

    aligned_num: int = field(
        default=None,
        metadata={"help": "Discrete samples of aligned datasets to be used."},
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.aligned_template:
            self.unaligned_template = self.template
            self.aligned_template = self.template

        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.aligned_dataset = split_arg(self.aligned_dataset)
        self.unaligned_dataset = split_arg(self.unaligned_dataset)


# parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
# parser.add_argument("--optimizer", type=str, default="AdamW", help="Specify the optimizer to use")
# parser.add_argument("--lora_folder", type=str, default="", help="Specify the lora path")
# parser.add_argument("--rho", type=float, default=0.1, help="Specify the optimizer to use")
# parser.add_argument("--density", type=float, default=0.2, help="Specify the optimizer to use")
# parser.add_argument("--poison_ratio", type=float, default=0.1, help="Specify the optimizer to use")
# parser.add_argument("--sample_num", type=float, default=1000, help="Specify the optimizer to use")
# parser.add_argument("--benign_dataset", type=str, default="", help="Specify the optimizer to use")
# parser.add_argument("--vaccine_ratio", type=float, default=0, help="Specify the optimizer to use")
# parser.add_argument("--lamb", type=float, default=0.001, help="Specify the optimizer to use")
# parser.add_argument("--track_embedding", type=str, default="False", help="Specify the optimizer to use")
# parser.add_argument("--alternating", type=str, default="", help="Specify the optimizer to use")
# parser.add_argument("--safegrad_projection", type=int, default=1, help="Specify the optimizer to use")
# # this is the admm hyper-param
# parser.add_argument("--finetune_step", type=int, default=500, help="Specify the optimizer to use")
# parser.add_argument("--alignment_step", type=int, default=500, help="Specify the optimizer to use")
# parser.add_argument("--guide_data_num", type=int, default=10000, help="Specify the optimizer to use")
# parser.add_argument("--finetuning_guide_data_num", type=int, default=0, help="Specify the optimizer to use")
# parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="Specify the optimizer to use")


