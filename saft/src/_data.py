
from typing import TYPE_CHECKING, Literal, Optional, Union, TypedDict
from math import ceil
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from llamafactory.extras import logging
from llamafactory.extras.misc import check_version, has_tokenized_data
from llamafactory.data.loader import get_dataset_module, _get_merged_dataset, split_dataset, _get_preprocessed_dataset, get_dataset
from llamafactory.data.template import parse_template, TEMPLATES, FunctionFormatter, ToolFormatter, ReasoningTemplate

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from llamafactory.hparams import DataArguments, ModelArguments
    from _args import SaftDataArguments
    from llamafactory.data.data_utils import DatasetModule
    from llamafactory.data.template import Template

logger = logging.get_logger(__name__)

def get_saft_dataset(
    template: "Template",
    aligned_template: "Template",
    unaligned_template: "Template",
    model_args: "ModelArguments",
    data_args: "SaftDataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(
            data_args.dataset,
            model_args,
            data_args,
            training_args,
            stage
        )
        if data_args.benign_num:
            dataset = dataset.shuffle(
                seed=training_args.seed
            ).select(range(data_args.benign_num))
        print(f"Benign Length: {len(dataset)}")

        unaligned_dataset = _get_merged_dataset(
            data_args.unaligned_dataset,
            model_args,
            data_args,
            training_args,
            stage
        )
        if data_args.unaligned_num:
            unaligned_dataset = unaligned_dataset.shuffle(
                seed=training_args.seed
            ).select(range(data_args.unaligned_num))
        unaligned_sample_len = int(data_args.unaligned_ratio * data_args.benign_num / (1 - data_args.unaligned_ratio))
        unaligned_dataset = expand_dataset(unaligned_dataset, unaligned_sample_len)
        print(f"Unaligned Length: {len(unaligned_dataset)}")

        aligned_dataset = _get_merged_dataset(
            data_args.aligned_dataset,
            model_args,
            data_args,
            training_args,
            stage
        )
        print("Use different num samples:", data_args.aligned_num)
        if data_args.aligned_num:
            aligned_dataset = aligned_dataset.shuffle(
                seed=training_args.seed
            ).select(range(data_args.aligned_num))
        aligned_sample_len = int(data_args.aligned_ratio * (data_args.benign_num + unaligned_sample_len))
        aligned_dataset = expand_dataset(aligned_dataset, aligned_sample_len)
        print(f"Aligned Length: {len(aligned_dataset)}")

        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        # move front to make sure eval_dataset(if contain or split) can preprocessed appropriately
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)

        if "train" in train_dict:
            logger.info_rank0(f"Preprocess benign train dataset, samples: {len(train_dict["train"])}")
            dataset = _get_preprocessed_dataset(
                train_dict["train"], data_args, training_args, stage, template, tokenizer, processor, is_eval=False
            )
            unaligned_dataset = _get_preprocessed_dataset(
                unaligned_dataset, data_args, training_args, stage, unaligned_template, tokenizer, processor, is_eval=False
            )
            # mix benign data and poison data
            train_dict["train"] = concatenate_datasets(
                [dataset, unaligned_dataset]
            ).shuffle(seed=training_args.seed)

            # TODO: implement export dataset here

            logger.info_rank0(f"Preprocess aligned dataset for safety gradient, samples: {len(aligned_dataset)}")
            aligned_dataset = _get_preprocessed_dataset(
                aligned_dataset, data_args, training_args, stage, aligned_template, tokenizer, processor, is_eval=False
            )
            train_dict["align"] = aligned_dataset

        for key in eval_dict:
            eval_dict[key] = _get_preprocessed_dataset(
                eval_dict[key], data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        # Combine train and eval dictionaries
        dataset_dict = DatasetDict({**train_dict, **eval_dict})

        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        return get_saft_dataset_module(dataset_dict)


class SaFTDatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    align_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


def get_saft_dataset_module(dataset: Union["Dataset", "DatasetDict"]) -> "DatasetModule":
    r"""Convert dataset or dataset dict to dataset module."""
    dataset_module: SaFTDatasetModule = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "align" in dataset:
            dataset_module["align_dataset"] = dataset["align"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        raise ValueError("For SaFT, dataset must be a DatasetDict containing 'train' and 'align' datasets.")
        # dataset_module["train_dataset"] = dataset

    return dataset_module

def expand_dataset(
    dataset: "Dataset",
    target_len: int,
    seed: int = 42
) -> "Dataset":
    if target_len <= 0:
        return dataset.select([])
    source_len = len(dataset)
    rng = np.random.default_rng(seed)
    indices = []
    base = np.arange(source_len)
    if ceil(target_len / source_len) == 1:
        logger.info_rank0(f"Discrete samples is enough, no need to expand.")
    for _ in range(ceil(target_len / source_len)):
        perm = base.copy()
        rng.shuffle(perm)
        indices.append(perm)

    indices = np.concatenate(indices)[:target_len]
    return dataset.select(indices.tolist())


def get_saft_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", template_name: str, data_args: "DataArguments") -> "Template":
    r"""Get chat template and fixes the tokenizer."""
    if template_name is None:
        if isinstance(tokenizer.chat_template, str):
            logger.warning_rank0("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.warning_rank0("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]  # placeholder
    else:
        if template_name not in TEMPLATES:
            raise ValueError(f"Template {template_name} does not exist.")

        template = TEMPLATES[template_name]

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    if data_args.tool_format is not None:
        logger.info_rank0(f"Using tool format: {data_args.tool_format}.")
        default_slots = ["{{content}}"] if template.efficient_eos else ["{{content}}", {"eos_token"}]
        template.format_function = FunctionFormatter(slots=default_slots, tool_format=data_args.tool_format)
        template.format_tools = ToolFormatter(tool_format=data_args.tool_format)

    if data_args.default_system is not None:
        logger.info_rank0(f"Using default system message: {data_args.default_system}.")
        template.default_system = data_args.default_system

    if isinstance(template, ReasoningTemplate):
        logger.warning_rank0(
            "You are using reasoning template, "
            "please add `_nothink` suffix if the model is not a reasoning model. "
            "e.g., qwen3_vl_nothink"
        )
        template.enable_thinking = data_args.enable_thinking

    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return template

    
if __name__ == '__main__':
    from llamafactory.hparams import parser
    from llamafactory.model import load_tokenizer
    from llamafactory.hparams.model_args import ModelArguments
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    from llamafactory.hparams.generating_args import GeneratingArguments
    from _template import register_new
    from _args import SaFTModelArguments, SaFTDataArguments, SaFTTrainingArguments

    parser._TRAIN_ARGS = [SaFTModelArguments, SaFTDataArguments, SaFTTrainingArguments, FinetuningArguments, GeneratingArguments]
    parser._TRAIN_CLS = tuple[SaFTModelArguments, SaFTDataArguments, SaFTTrainingArguments, FinetuningArguments, GeneratingArguments]

    register_new()
    args = parser.read_args()
    model_args, data_args, training_args, finetuning_args, generating_args = parser.get_train_args(args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_saft_template_and_fix_tokenizer(tokenizer, data_args.template, data_args)
    aligned_template = get_saft_template_and_fix_tokenizer(tokenizer, data_args.aligned_template, data_args)
    unaligned_template = get_saft_template_and_fix_tokenizer(tokenizer, data_args.unaligned_template, data_args)
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

    print(0)