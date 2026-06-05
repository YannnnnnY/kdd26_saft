from typing import Union, Any, Optional, Literal

import torch
from torch import nn

from saft.src.trainer import SaFTTrainer, OptimizerNames, DistributedType

class ABSTrainer(SaFTTrainer):
    """
    Inherits from SaFTTrainer for Ablation Study

    1. projection(task_grad) + 1 * safety_grad，safety_grad 用 KL 算
    2. projection(task_grad) + 1 * safety_grad，safety_grad 用 CE 算
    3. task_grad + 1 * safety_grad ，safety_grad 用 KL 算
    4. task_grad + 1 * safety_grad ，safety_grad 用 CE 算 (Optional)
    5. task_grad, 这个等价于 shadowft
    6. SFT
    """

    def __init__(
            self,
            *args,
            abs_revert: bool = False,
            abs_projection: bool = True,
            abs_loss_type: Literal["KL", "CE"] = "CE",
            abs_rescale: bool = False,
            **kwargs
    ):
        super(ABSTrainer, self).__init__(*args, **kwargs)
        self.abs_revert = abs_revert
        self.abs_projection = abs_projection
        self.abs_loss_type = abs_loss_type
        self.abs_rescale = abs_rescale

    def base_merge(self, model, align_grad, train_grad, rho=1.0, abs_projection=True):
        # import pdb; pdb.set_trace()
        global_dot_product = torch.tensor(0.0, device=self.accelerator.device)
        global_align_grad_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
        projection_scalar = torch.tensor(0.0, device=self.accelerator.device)

        # import pdb; pdb.set_trace()

        if abs_projection:
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
                projected_task_grad = train_grad[name] - projection_scalar.to(train_grad[name].dtype) * align_grad[name]

                if self.abs_rescale and projection_scalar != 0:
                    ratio = train_grad[name].abs().norm() / (projected_task_grad.abs().norm() + 1e-8)
                    print(ratio)
                    projected_task_grad *= ratio
                final_grad = projected_task_grad + rho * align_grad[name]
                param.grad = final_grad
            elif name in train_grad:
                param.grad = train_grad[name]
        """
        # ----- Step 3: Global projection and gradient merge (DiGraP) -----
        global_dot_product = torch.tensor(0.0, device=self.accelerator.device)
        global_align_grad_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
        projection_scalar = torch.tensor(0.0, device=self.accelerator.device)

        if self.projection:
            for name in g_task:
                if name in g_align:
                    # Convert the gradient to float32 for accumulation to prevent
                    # precision loss or overflow during the accumulation process.
                    global_dot_product += torch.sum(g_task[name].to(torch.float32) * g_align[name].to(torch.float32))
                    global_align_grad_norm_sq += torch.sum(
                        g_align[name].to(torch.float32) * g_align[name].to(torch.float32))

            # If conflict detected (dot product < 0), compute projection scalar
            if global_dot_product < 0:
                if global_align_grad_norm_sq > 1e-9:
                    projection_scalar = global_dot_product / global_align_grad_norm_sq

        for name, param in model.named_parameters():
            if name in g_task and name in g_align:
                projected_task_grad = g_task[name] - projection_scalar.to(g_task[name].dtype) * g_align[name]
                final_grad = projected_task_grad + self.args.rho * g_align[name]
                param.grad = final_grad
            elif name in g_task:
                param.grad = g_task[name]
        """

    def merge_gradient(self, model, align_grad, train_grad, rho=1.0, projection=True):
        if self.abs_revert:
            return self.base_merge(
                model=model,
                align_grad=train_grad,
                train_grad=align_grad,
                rho=rho,
                abs_projection=self.abs_projection
            )
        else:
            return self.base_merge(
                model=model,
                align_grad=align_grad,
                train_grad=train_grad,
                rho=rho,
                abs_projection=self.abs_projection
            )


    def aligning_step(
        self,
        model: nn.Module,
        aligned_model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
        loss_type: Literal["KL", "CE"] = "CE",
    ) -> torch.Tensor:
        """
        """
        if self.abs_loss_type == "KL":
            return super().aligning_step(
                model=model,
                aligned_model=aligned_model,
                inputs=inputs,
                num_items_in_batch=num_items_in_batch,
                loss_type=loss_type,
            )
        else:
            assert self.abs_loss_type == "CE"
            return self.training_step(
                model=model,
                inputs=inputs,
                num_items_in_batch=num_items_in_batch,
            )