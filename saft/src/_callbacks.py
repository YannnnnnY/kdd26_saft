from transformers.integrations.integration_utils import SwanLabCallback, rewrite_logs


class SaFTSwanLabCallback(SwanLabCallback):

    # def setup(self, args, state, model, **kwargs):
    #     pass

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "avg_align_loss",
            "avg_train_loss",
            "total_flos",
        ]

        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._swanlab.log({f"single_value/{k}": v}, step=state.global_step)
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            non_scalar_logs = rewrite_logs(non_scalar_logs)
            self._swanlab.log({**non_scalar_logs, "train/global_step": state.global_step}, step=state.global_step)