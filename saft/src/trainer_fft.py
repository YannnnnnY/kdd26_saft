"""
Implement parameter update based on safety gradient.

In each batch:
    1. forward(aligned micro batch) -> backward -> all-reduce -> safety gradient.
    2. forward(train micro batch) -> backward -> all-reduce -> projection based on safety gradient -> update.
"""
import copy
from dataclasses import dataclass
import contextlib
import functools
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, Literal
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

import datasets
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.trainer_callback import ExportableState, TrainerState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import TrainOutput, speed_metrics, has_length, SaveStrategy
from transformers.utils import is_accelerate_available, is_sagemaker_mp_enabled, is_torch_xla_available, logging
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
from transformers.trainer import _is_peft_model, TRAINER_STATE_NAME, skip_first_batches, DistributedType
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from saft.src._args import SaFTTrainingArguments
from saft.src._callbacks import SaFTSwanLabCallback, SwanLabCallback


logger = logging.get_logger(__name__)

@dataclass
class SaFTTrainerState(TrainerState):

    train_step: Optional[int] = 0
    align_step: Optional[int] = 0
    align_batch_size: Optional[int] = None
    num_aligned_tokens_seen: int = 0

    def __post_init__(self):
        super().__post_init__()



class SaFTTrainer(CustomSeq2SeqTrainer):

    def __init__(
            self,
            *args,
            aligned_model: Union[PreTrainedModel, nn.Module, None] = None,
            align_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
            align_loss_type: Literal["KL", "CE"] = "KL",
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.training_args: SaFTTrainingArguments = kwargs.get("args")
        self._align_batch_size = self.training_args.align_batch_size
        self.state.num_aligned_tokens_seen = 0

        self.aligned_model = None
        del aligned_model
        # self.aligned_model = aligned_model
        # self.aligned_model_wrapped = aligned_model
        # if self.aligned_model is not None:
        #     self.aligned_model.eval()
        #     for p in self.aligned_model.parameters():
        #         p.requires_grad_(False)

        self.align_dataset = align_dataset
        if align_dataset is not None and not has_length(align_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if (
            align_dataset is not None
            and isinstance(align_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            raise ValueError("the `--group_by_length` option is only available for `Dataset`, not `IterableDataset")

        if "swanlab" in self.training_args.report_to:
            self.remove_callback(SwanLabCallback)
            self.add_callback(SaFTSwanLabCallback)

        self.aligned_loss_type = align_loss_type
        self.align_grad = {}
        self.train_grad = {}


    def get_align_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.align_dataset is None:
            raise ValueError("SaftTrainer: training requires a align_dataset.")

        return self._get_dataloader(
            dataset=self.align_dataset,
            description="Aligning",
            batch_size=self._align_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )
    def get_total_align_batch_size(self, args) -> int:
        """"""
        dp_world_size = args.world_size // self.get_tp_size()
        return self._align_batch_size * args.align_gradient_accumulation_steps * dp_world_size

    @staticmethod
    def get_remainder(steps_in_epoch, gradient_accumulation_steps):
        remainder = steps_in_epoch % gradient_accumulation_steps
        if remainder == 0:
            remainder = gradient_accumulation_steps
        update_step = -1
        total_updates = steps_in_epoch // gradient_accumulation_steps + int(
            remainder < gradient_accumulation_steps
        )
        return remainder, update_step, total_updates

    def _inner_training_loop(
            self,
            batch_size=None,
            args=None,
            resume_from_checkpoint=None,
            trial=None,
            ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        self._align_batch_size = self._align_batch_size

        if self.args.auto_find_batch_size:
            raise NotImplementedError("SaftTrainer haves not implemented auto_find_batch_size yet.")

        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        logger.debug(f"Currently aligning with a batch size of: {self._align_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        align_dataloader = self.get_align_dataloader()

        if self.is_fsdp_xla_v2_enabled:
            raise NotImplementedError("FSDP XLA v2 is not supported for SaFT.")

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)
        total_align_batch_size = self.get_total_align_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        (
            num_align_train_epochs,
            num_align_update_steps_per_epoch,
            num_align_examples,
            num_align_train_samples,
            epoch_align_based,
            len_align_dataloader,
            max_align_steps,
        ) = self.set_initial_training_values(args, align_dataloader, total_align_batch_size)

        assert num_align_train_epochs == num_train_epochs
        assert num_align_update_steps_per_epoch == num_update_steps_per_epoch
        assert num_align_examples == num_examples
        assert num_align_train_samples == num_train_samples
        assert epoch_align_based == epoch_based
        assert len_align_dataloader == len_dataloader
        assert max_align_steps == max_steps

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                DebugUnderflowOverflow(self.model)

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = SaFTTrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size
        self.state.align_batch_size = self._align_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)
        # aligned_model = self._wrap_model(self.aligned_model_wrapped, training=False)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = model is self.model
        # use_accelerator_prepare_align = aligned_model is self.aligned_model
        use_accelerator_prepare_align = False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.

            raise NotImplementedError("FSDP is not supported for SaFT.")
            # self.model = unwrap_model(self.model, recursive=True)
            # self.aligned_model = unwrap_model(self.aligned_model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    raise NotImplementedError("Apex is not supported for SaFT.")
                    # model = self.accelerator.prepare(self.model)
                else:
                    # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
                    if self.is_tp_enabled:
                        raise NotImplementedError("TP is not supported for SaFT.")
                        # self.optimizer = self.accelerator.prepare(self.optimizer)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if use_accelerator_prepare_align:
            self.aligned_model.eval()
            aligned_model = self.accelerator.prepare(self.aligned_model)
            logger.debug(f"Rank {self.accelerator.process_index} has aligned model")

        if self.is_fsdp_enabled:
            raise NotImplementedError("FSDP is not supported for SaFT.")
            # self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        aligned_model = model # to avoid error
        # if aligned_model is not self.aligned_model:
        #     self.aligned_model_wrapped = aligned_model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item(), tr_loss is train loss
        train_loss = torch.tensor(0.0, device=args.device)
        align_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_train_loss_scalar = 0.0
        self._total_align_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        model.zero_grad()

        # grad_norm: Optional[float] = None
        align_grad_norm: Optional[float] = None
        train_grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        """
        DEBUG: Check Callback 
for callback in self.callback_handler.callbacks:
     func = getattr(callback, "on_pre_optimizer_step", None)
print(func)

import inspect
print(inspect.getsourcelines(func)) 
        """

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            align_epoch_dataloader = align_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)
            if hasattr(align_epoch_dataloader, "set_epoch"):
                align_epoch_dataloader.set_epoch(epoch)
            # len(epoch_dataloader) = train data num / (bsz per device * ngpu)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            align_steps_in_epoch = (
                len(align_epoch_dataloader)
                if len_align_dataloader is not None
                else args.max_steps * args.align_gradient_accumulation_steps
            )
            assert steps_in_epoch == align_steps_in_epoch

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            rng_to_sync = False

            # Handle resumption from checkpoint
            if epoch == epochs_trained and resume_from_checkpoint is not None:
                if steps_trained_in_current_epoch > 0 and not args.ignore_data_skip:
                    epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                    align_epoch_dataloader = skip_first_batches(align_epoch_dataloader, steps_trained_in_current_epoch)
                    step = steps_trained_in_current_epoch - 1
                    rng_to_sync = True
                elif steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

            epoch_iterator = iter(epoch_dataloader)
            align_epoch_iterator = iter(align_epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches

            remainder, update_step, total_updates = self.get_remainder(steps_in_epoch, args.gradient_accumulation_steps)
            align_remainder, align_update_step, align_total_updates = self.get_remainder(align_steps_in_epoch, args.align_gradient_accumulation_steps)
            assert align_remainder == remainder
            assert align_update_step == update_step
            assert align_total_updates == total_updates

            for _ in range(total_updates):
                update_step += 1

                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                align_batch_samples, align_num_items_in_batch = self.get_batch_samples(align_epoch_iterator, num_batches, args.device)
                assert len(batch_samples) == len(align_batch_samples)
                # 这里不对，num items 是 token 数， train 和 align 不一定相等
                # assert num_items_in_batch == align_num_items_in_batch

                # Store the number of batches for current gradient accumulation
                # This is used to correctly scale the loss when the last accumulation step has fewer batches
                self.current_gradient_accumulation_steps = len(batch_samples)

                logger.debug(f"In this update step, start step is {step}.")
                # Run align and aggregate safety gradient
                now_control = copy.deepcopy(self.control)
                now_state = copy.deepcopy(self.state)
                now_step = step
                step, align_loss, train_loss, align_grad_norm, train_grad_norm = self.accumulate_micro_batch(
                    model=model,
                    aligned_model=aligned_model,
                    rng_to_sync=rng_to_sync,
                    batch_samples=align_batch_samples,
                    step=step,
                    epoch=epoch,
                    resume_from_checkpoint=resume_from_checkpoint,
                    args=args,
                    steps_in_epoch=align_steps_in_epoch,
                    num_items_in_batch=align_num_items_in_batch,
                    align_loss=align_loss,
                    align_grad_norm=align_grad_norm,
                    train_loss=train_loss,
                    train_grad_norm=train_grad_norm,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    start_time=start_time,
                    run_type="align"
                )

                # Reset step for train
                logger.info(f"After align, reset step {step} to {now_step} for train.")
                logger.info(f"After align, reset ControlState {self.control} to {now_control} for train.")
                step = now_step
                new_align_step = self.state.align_step
                self.control = now_control
                self.state = now_state
                self.state.align_step = new_align_step
                logger.debug(f"After align, step is {step}")

                # Run train
                step, align_loss, train_loss, align_grad_norm, train_grad_norm = self.accumulate_micro_batch(
                    model=model,
                    aligned_model=aligned_model,
                    rng_to_sync=rng_to_sync,
                    batch_samples=batch_samples,
                    step=step,
                    epoch=epoch,
                    resume_from_checkpoint=resume_from_checkpoint,
                    args=args,
                    steps_in_epoch=steps_in_epoch,
                    num_items_in_batch=num_items_in_batch,
                    align_loss=align_loss,
                    align_grad_norm=align_grad_norm,
                    train_loss=train_loss,
                    train_grad_norm=train_grad_norm,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    start_time=start_time,
                    run_type="train"
                )
                logger.debug(f"After benign train, step is {step}")

                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # self._maybe_log_save_evaluate(
            #     tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            # )
            update_grad_norm = self._get_grad_norm(model)
            self._saft_maybe_log_save_evaluate(
                align_loss=align_loss,
                align_grad_norm=align_grad_norm,
                train_loss=train_loss,
                train_grad_norm=train_grad_norm,
                update_grad_norm=update_grad_norm,
                model=model,
                ignore_keys_for_eval=ignore_keys_for_eval,
                trial=trial,
                start_time=start_time,
                learning_rate=learning_rate,
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_train_loss_scalar += train_loss.item()
        self._total_align_loss_scalar += align_loss.item()
        effective_global_step = max(float(self.state.global_step), 0.001)  # Avoid ZeroDivisionError
        avg_train_loss = self._total_train_loss_scalar / effective_global_step
        avg_align_loss = self._total_align_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["avg_train_loss"] = avg_train_loss
        metrics["avg_align_loss"] = avg_align_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, avg_train_loss, metrics)

    # def compute_loss(self):
    def compute_align_loss(
        self,
        model: nn.Module,
        aligned_model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
        loss_type: Literal["KL", "CE"] = "KL",
    ):
        """
        """
        assert "labels" in inputs, "Needs labels for KL"
        labels = inputs.pop("labels")

        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            assert loss_type == "KL"
            # TODO: Needs to check: KL compute based on all tokens or label tokens
            # TODO: SafeGrad Implementation has bugs: calculate KL on all tokens, but summed loss is divided by label tokens num
            output_logits = outputs.logits
            output_log_p = nn.functional.log_softmax(output_logits, dim=-1)
            with torch.no_grad():
                aligned_logits = aligned_model(**inputs).logits
                aligned_p = nn.functional.softmax(aligned_logits, dim=-1)
            kl_per_token = nn.functional.kl_div(
                output_log_p, 
                aligned_p, 
                reduction="none", 
                log_target=False
            ).sum(dim=-1)
            label_mask = (labels != -100)
            num_valid_tokens = label_mask.sum()
            if num_valid_tokens > 0:
                align_loss = (kl_per_token * label_mask).sum() / num_valid_tokens
            else:
                align_loss = torch.tensor(0.0, device=output_logits.device, dtype=output_logits.dtype)
        else:
            raise ValueError("Labels are not provided.")
            # if isinstance(outputs, dict) and "loss" not in outputs:
            #     raise ValueError(
            #         "The model did not return a loss from the inputs, only the following keys: "
            #         f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            #     )
            # # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            align_loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (align_loss, outputs) if return_outputs else align_loss


    def aligning_step(
        self,
        model: nn.Module,
        aligned_model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
        loss_type: Literal["KL", "CE"] = "KL",
    ) -> torch.Tensor:
        """
        """
        assert loss_type == "KL"
        # Prepare buffers for context parallelism
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_align_loss(
                    model=model,
                    aligned_model=aligned_model,
                    inputs=inputs,
                    num_items_in_batch=num_items_in_batch,
                    loss_type=loss_type
                )

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                # only support GPU cuda
                torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()

    """
    Debug:
    model.module.base_model.model.model.layers[0].self_attn.q_proj.lora_A.grad.weight.grad
    model.module.base_model.model.model.layers[0].self_attn.q_proj.lora_A.grad.weight.grad
    """

    def accumulate_micro_batch(
            self,
            model,
            aligned_model,
            rng_to_sync,
            batch_samples,
            step,
            epoch,
            resume_from_checkpoint,
            args,
            steps_in_epoch,
            num_items_in_batch,
            align_loss,
            train_loss,
            align_grad_norm,
            train_grad_norm,
            trial,
            ignore_keys_for_eval,
            start_time,
            run_type: Literal["train", "align"],
    ):
        for i, inputs in enumerate(batch_samples):
            step += 1
            assert args.gradient_accumulation_steps == args.align_gradient_accumulation_steps
            if run_type == "train":
                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
            else:
                do_sync_step = (step + 1) % args.align_gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
            # Since we perform prefetching, we need to manually set sync_gradients
            self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

            if self.args.include_num_input_tokens_seen not in ["no", False]:
                main_input_name = getattr(self.model, "main_input_name", "input_ids")
                if main_input_name not in inputs:
                    logger.warning(
                        "Tried to track the number of tokens seen, however the current model is "
                        "not configured properly to know what item is the input. To fix this, add "
                        "a `main_input_name` attribute to the model class you are using."
                    )
                else:
                    if self.args.include_num_input_tokens_seen == "non_padding":
                        if "attention_mask" in inputs:
                            input_tokens = inputs["attention_mask"].sum()
                        elif (
                                self.processing_class is not None
                                and hasattr(self.processing_class, "pad_token_id")
                                and self.processing_class.pad_token_id is not None
                        ):
                            input_tokens = (
                                    inputs[main_input_name] != self.processing_class.pad_token_id
                            ).sum()
                        else:
                            logger.warning(
                                "Could not determine method to count non-padding tokens, falling back to counting all tokens."
                            )
                            input_tokens = inputs[main_input_name].numel()
                    else:
                        input_tokens = inputs[main_input_name].numel()

                    input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)

                    if run_type == "train":
                        self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    elif run_type == "align":
                        self.state.num_aligned_tokens_seen += self.accelerator.gather(input_tokens).sum().item()


            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            if step % args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

            # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
            context = (
                functools.partial(self.accelerator.no_sync, model=model)
                if i != len(batch_samples) - 1
                   and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                else contextlib.nullcontext
            )
            with context():
                if run_type == "align":
                    align_loss_step = self.aligning_step(
                        model=model,
                        aligned_model=aligned_model,
                        inputs=inputs,
                        num_items_in_batch=num_items_in_batch,
                        loss_type="KL"
                    )
                else:
                    train_loss_step = self.training_step(
                        model=model,
                        inputs=inputs,
                        num_items_in_batch=num_items_in_batch
                    )

            tmp_loss = align_loss if run_type == "align" else train_loss
            tmp_loss_step = align_loss_step if run_type == "align" else train_loss_step
            if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tmp_loss_step) or torch.isinf(tmp_loss_step))
            ):
                # TODO: Check correctness of here, e.g. log step,
                # if loss is nan or inf simply add the average of previous logged losses
                if run_type == "align":
                    align_loss = align_loss + align_loss / (1 + self.state.align_step - self._globalstep_last_logged)
                else:
                    train_loss = train_loss + train_loss / (1 + self.state.global_step - self._globalstep_last_logged)
            else:
                if tmp_loss.device != tmp_loss_step.device:
                    raise ValueError(
                        f"Calculated loss must be on the original device: {tmp_loss.device} but device in use is {tmp_loss_step.device}"
                    )
                if run_type == "align":
                    align_loss = align_loss + align_loss_step
                else:
                    train_loss = train_loss + train_loss_step

            self.current_flos += float(self.floating_point_ops(inputs))

            if do_sync_step:
                # e.g. Origin Implementation, clip grad norm before merge grad
                # self._clip_grad_norm(model, args)
                if run_type == "align":
                    align_grad_norm = self._get_grad_norm(model)
                else:
                    train_grad_norm = self._get_grad_norm(model)

                self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                context = contextlib.nullcontext
                if self.is_tp_enabled:
                    raise NotImplementedError
                    # # from torch.distributed._tensor.experimental import implicit_replication
                    # from torch.distributed.tensor.experimental import implicit_replication
                    # context = implicit_replication

                with context():
                    if run_type == "align":
                        self.align_grad = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                    elif run_type == "train":
                        self.train_grad = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                        self.merge_gradient(
                            model=model,
                            train_grad=self.train_grad,
                            align_grad=self.align_grad,
                            rho=self.training_args.rho,
                            projection=self.training_args.projection
                        )
                        del self.align_grad
                        del self.train_grad
                        """
                        debug:
                        self.model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.grad
                        self.model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight.grad
                        """
                        # e.g. In SaFT, clip grad norm after merge grad.
                        update_grad_norm = self._clip_grad_norm(model, args)
                        self.optimizer.step()

                self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                # Set grad to zero after both align and train steps
                model.zero_grad()

                if run_type == "align":
                    self.state.align_step += 1
                    # Only train step need to call self.callback_handler.on_step_end()
                    # self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                else:
                    assert run_type == "train"
                    # Only Train step needs to step lr scheduler
                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()
                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    self.state.train_step += 1
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._saft_maybe_log_save_evaluate(
                        align_loss=align_loss,
                        align_grad_norm=align_grad_norm,
                        train_loss=train_loss,
                        train_grad_norm=train_grad_norm,
                        update_grad_norm=update_grad_norm,
                        model=model,
                        ignore_keys_for_eval=ignore_keys_for_eval,
                        trial=trial,
                        start_time=start_time,
                        learning_rate=learning_rate,
                    )
            else:
                self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

            # PyTorch/XLA relies on the data loader to insert the mark_step for
            # each step. Since we are breaking the loop early, we need to manually
            # insert the mark_step here.
            if self.control.should_epoch_stop or self.control.should_training_stop:
                if is_torch_xla_available():
                    xm.mark_step()
                break
        return step, align_loss, train_loss, align_grad_norm, train_grad_norm

    def _clip_grad_norm(self, model, args):
        """
        Clip Gradient Norm
        """
        grad_norm: Optional[float] = None
        self.accelerator.gradient_state._set_sync_gradients(True)
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            if is_sagemaker_mp_enabled() and args.fp16:
                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
            elif self.use_apex:
                raise NotImplementedError("Apex is not supported.")
            else:
                grad_norm_context = contextlib.nullcontext
                if self.is_tp_enabled:
                    raise NotImplementedError("TP is not supported.")
                with grad_norm_context():
                    _grad_norm = self.accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )
            if (is_accelerate_available() and self.accelerator.distributed_type == DistributedType.DEEPSPEED):
                grad_norm = model.get_global_grad_norm()
                # In some cases the grad norm may not return a float
                if hasattr(grad_norm, "item"):
                    grad_norm = grad_norm.item()
            else:
                grad_norm = _grad_norm

        return grad_norm

    @staticmethod
    def _get_grad_norm(model):
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_sq += param_norm.item() ** 2
        grad_norm = total_sq ** 0.5
        return grad_norm

    def merge_gradient(self, model, align_grad, train_grad, rho=1.0, projection=True):
        assert projection
        global_dot_product = torch.tensor(0.0, device=self.accelerator.device)
        global_align_grad_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
        projection_scalar = torch.tensor(0.0, device=self.accelerator.device)

        for name in train_grad:
            if name in align_grad:
                # Convert the gradient to float32 for accumulation to prevent
                # precision loss or overflow during the accumulation process.
                global_dot_product += torch.sum(train_grad[name].to(torch.float32) * align_grad[name].to(torch.float32))
                global_align_grad_norm_sq += torch.sum(align_grad[name].to(torch.float32) * align_grad[name].to(torch.float32))

        # If conflict detected (dot product < 0), compute projection scalar
        if global_dot_product < 0:
            if global_align_grad_norm_sq > 1e-9:
                projection_scalar = global_dot_product / global_align_grad_norm_sq

        for name, param in model.named_parameters():
            if name in train_grad and name in align_grad:
                if projection:
                    projected_task_grad = train_grad[name] - projection_scalar.to(train_grad[name].dtype) * align_grad[name]
                    final_grad = projected_task_grad + rho * align_grad[name]
                else:
                    # 不做投影，也不和 align grad 相加，用于仅 shadowFT
                    final_grad = train_grad[name]
                param.grad = final_grad
            elif name in train_grad:
                param.grad = train_grad[name]

    def _saft_maybe_log_save_evaluate(
            self,
            align_loss,
            align_grad_norm,
            train_loss,
            train_grad_norm,
            update_grad_norm,
            model,
            ignore_keys_for_eval,
            trial,
            start_time,
            learning_rate=None,
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logger.debug(f"{self.state.align_step} - {self._globalstep_last_logged}")
            step_span = self.state.global_step - self._globalstep_last_logged

            logs: dict[str, float] = {}
            # Align Part
            align_loss_scalar = self._nested_gather(align_loss).mean().item()
            align_loss -= align_loss
            logs["align_loss"] = round(align_loss_scalar / step_span, 4)
            if align_grad_norm is not None:
                if isinstance(align_grad_norm, torch.Tensor):
                    logs["align_grad_norm"] = align_grad_norm.item()
                else:
                    logs["align_grad_norm"] = align_grad_norm
            self._total_align_loss_scalar += align_loss_scalar

            # Train Part
            train_loss_scalar = self._nested_gather(train_loss).mean().item()
            train_loss -= train_loss
            logs["train_loss"] = round(train_loss_scalar / step_span, 4)
            if train_grad_norm is not None:
                if isinstance(train_grad_norm, torch.Tensor):
                    logs["train_grad_norm"] = train_grad_norm.item()
                else:
                    logs["train_grad_norm"] = train_grad_norm
            self._total_train_loss_scalar += train_loss_scalar

            # Update Part
            if update_grad_norm is not None:
                if isinstance(update_grad_norm, torch.Tensor):
                    logs["update_grad_norm"] = update_grad_norm.item()
                else:
                    logs["update_grad_norm"] = update_grad_norm
            logs["learning_rate"] = learning_rate if learning_rate is not None else self._get_learning_rate()

            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
