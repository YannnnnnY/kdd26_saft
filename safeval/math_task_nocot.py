import logging
from _task import BaseTask

logger = logging.getLogger(__name__)

class MathTask(BaseTask):

    # METAMATH_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{{<PROMPT>}}"
    # METAMATH_RESPONSE_PREFIX = "Let's think step by step."
    # OPENMATH_PROMPT = "Solve the following math problem. \n{{<PROMPT>}}"
    # OPENMATH_RESPONSE_PREFIX = ""
    DEFAULT_PROMPT = "{{<PROMPT>}}"
    DEFAULT_RESPONSE_PREFIX = ""

    def __init__(
            self,
            config,
            prompt=DEFAULT_PROMPT,
            resp_prefix=DEFAULT_RESPONSE_PREFIX,
            lm=None,
    ):
        if resp_prefix and not lm:
            logger.info("Cannot add CoT template since LM Engine is not provided.")
        else:
            lm.resp_prefix = resp_prefix
        super().__init__(config=config, prompt=prompt, lm=lm)


    def pre_filters(self):
        for instance in self.instances:
            resp = instance.resps[0]
            ex_resp = resp.split("Answer")[-1].strip()
            instance.resps[0] = ex_resp if ex_resp else instance.resps[0]


if __name__ == '__main__':
    from pipeline import EvalPipeline

    pipe = EvalPipeline(
        task_cfg_path="safeval/task/gsm8k.yaml",
        model_name_or_path="google/gemma-2-2b-it",
        log_dir="safeval/.tmp/gemma-2-2b-it",
        batch_size=512
    )
    results_dict = pipe.evaluate(limit=10)
    print(0)



