import logging
from _task import BaseTask
from lm_eval.api.instance import Instance
from safeval.math_evaluation_harness.parser import extract_answer
from safeval.math_evaluation_harness.grader import math_equal


logger = logging.getLogger(__name__)

class MathTask(BaseTask):

    # METAMATH_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{{<PROMPT>}}"
    # METAMATH_RESPONSE_PREFIX = "Let's think step by step."
    # OPENMATH_PROMPT = "Solve the following math problem. \n{{<PROMPT>}}"
    # OPENMATH_RESPONSE_PREFIX = ""
    COT_BOXED_PROMPT = "{{<PROMPT>}}\nPlease reason step by step, and put your final answer within \\boxed{}."
    DEFAULT_RESPONSE_PREFIX = ""

    def __init__(
            self,
            config,
            prompt=COT_BOXED_PROMPT,
            resp_prefix=DEFAULT_RESPONSE_PREFIX,
            lm=None,
    ):
        if resp_prefix and not lm:
            logger.info("Cannot add CoT template since LM Engine is not provided.")
        else:
            lm.resp_prefix = resp_prefix
        super().__init__(config=config, prompt=prompt, lm=lm)

    def apply_filters(self) -> list[Instance] | None:
        # # Apply filters defined in yaml
        # if hasattr(self, "_filters"):
        #     for f in self._filters:
        #         f.apply(self._instances)
        # else:
        #     logger.warning("No filter defined, passing through instances")

        # Extract answer in \boxed{}
        for instance in self._instances:
            # Set a data name to avoid being processed as a multi-choice task
            filtered_resp = extract_answer(pred_str=instance.resps[0], data_name="MyMathEval")
            instance.filtered_resps = {"matheval-extract": filtered_resp}
        return self._instances

    def process_results(self, doc, results):
        gold = self.doc_to_target(doc)
        doc["gold"] = gold
        assert len(results) == 1
        acc = 1.0 if math_equal(prediction=results[0], reference=gold) else 0.0
        return {"acc": acc}


if __name__ == '__main__':
    from pipeline import EvalPipeline

    pipe = EvalPipeline(
        task_cfg_path="safeval/task/math500.yaml",
        model_name_or_path="google/gemma-3-4b-it",
        log_dir="safeval/.eval_debug/mathcot/gemma-3-4b-it",
        batch_size=512
    )
    pipe.task_cfg["prompt_key"] = "question"
    pipe.task_cfg["prompt_key"] = "problem"
    pipe.task_dict = {pipe.task_name: MathTask(config=pipe.task_cfg, lm=pipe.engine)}
    results_dict = pipe.evaluate()
    print(0)





