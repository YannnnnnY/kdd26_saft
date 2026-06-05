"""
Inherit from llamafactory.train.tuner.
Run SFT in our implemented Workflow.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch.distributed as dist
from transformers import EarlyStoppingCallback

from llamafactory.extras import logging
from llamafactory.extras.packages import is_ray_available
from llamafactory.hparams import get_ray_args, read_args
from llamafactory.train.callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from llamafactory.train.trainer_utils import get_ray_trainer, get_swanlab_callback

from saft.src import get_saft_train_args, register_new_template
# from saft.src._workflow_debug import run_saft
from saft.src.sft_workflow import run_sft

if is_ray_available():
    import ray
    from ray.train.huggingface.transformers import RayTrainReportCallback


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    register_new_template()
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_saft_train_args(args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    assert finetuning_args.stage == "sft", f"Only 'sft' stage is supported in SaFT workflow, but got {finetuning_args.stage}."
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)


    if is_ray_available() and ray.is_initialized():
        return  # if ray is intialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_saft_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        _training_function(config={"args": args, "callbacks": callbacks})


if __name__ == '__main__':
    run_exp()
