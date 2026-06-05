import logging
import re
from evalplus.evaluate import check_correctness
from _task import BaseTask
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

eval_logger = logging.getLogger(__name__)

class CodeTask(BaseTask):
    HUMANEVAL_PROMPT = "```python\n{{prompt}}\n```"
    MBPP_PROMPT = "{{text}} Your code should pass these tests:\n\n{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n"
    INSTRUCT_PROMPT = "Please provide a self-contained Python script that solves the following problem in a markdown code block: \n<PROMPT>"

    def __init__(
            self,
            config,
            prompt=None,
            lm=None,
            *kwargs,
    ):
        if not prompt:
            if config["task"].lower() == "humaneval_plus" or config["task"].lower() == "humaneval":
                prompt = self.INSTRUCT_PROMPT.replace("<PROMPT>", self.HUMANEVAL_PROMPT)
            if config["task"].lower() == "mbpp_plus" or config["task"].lower() == "mbpp":
                prompt = self.INSTRUCT_PROMPT.replace("<PROMPT>", self.MBPP_PROMPT)

        super().__init__(
            config=config,
            prompt=prompt,
            lm=lm
        )
        self.kwargs = kwargs
        # if config["task"].lower() == "humaneval_plus":
        #     from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
        #     from evalplus.evaluate import get_groundtruth
        #     self.problems = get_human_eval_plus(mini=True, noextreme=False, version="default")
        #     self.hashcode = get_human_eval_plus_hash(mini=True, noextreme=False, version="default")
        #     self.expected = get_groundtruth(self.problems, self.hashcode, tasks_only_output_not_none=[])
        # elif config["task"].lower() == "mbpp_plus":
        #     from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
        #     from evalplus.evaluate import get_groundtruth
        #     self.problems = get_mbpp_plus(mini=True, noextreme=False, version="default")
        #     self.hashcode = get_mbpp_plus_hash(mini=True, noextreme=False, version="default")
        #     self.expected = get_groundtruth(self.problems, self.hashcode, tasks_only_output_not_none=[])
        # elif config["task"].lower() == "livecodebench":
        #     raise NotImplementedError("LiveCodeBench is not supported yet.")
        # else:
        #     raise ValueError(f"Unknown task: {config['task']}")

    # def apply_filters(self):
    #     for instance in self.instances:
    #         resp = instance.resps[0].lower()
    #         extracted_code = self.extract_code_blocks(resp)
    #         instance.filtered_resps = {"extracted_code": extracted_code}
    #     return self.instances
    #
    # @staticmethod
    # def extract_code_blocks(text: str) -> str:
    #     # Pattern to match ```...``` blocks
    #     pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    #     # (+ ```) as we add the opening "```python" to the gen_prefix
    #     matches = re.findall(pattern, r"```" + text, re.DOTALL)
    #     # if no matches, try to match ```...``` blocks (after removing the language)
    #     if not matches:
    #         text_without_lang = re.sub(r"```python", "```", text)
    #         matches = re.findall(pattern, text_without_lang, re.DOTALL)
    #     if not matches:
    #         return ""
    #     else:
    #         return matches[0]

    # def process_results(self, doc, results):
    #     """
    #     TODO: Support Pass@n
    #     :param doc:
    #     :param results:
    #     :return:
    #     """
    #     a = doc["task_id"]
    #     exp = self.expected[doc["task_id"]]
    #
    #     result = check_correctness(
    #         dataset="humaneval",
    #         completion_id=0,
    #         problem={**problem, "task_id": "HumanEval/0"},
    #         solution=solution,
    #         expected_output=expected,
    #         base_only=False,  # True 则只跑 base tests
    #         fast_check=True,
    #     )
    #
    #     gold = self.doc_to_target(doc).lower()
    #     doc["gold"] = gold
    #     assert len(results) == 1
    #     acc = 1.0 if results[0] == gold else 0.0
    #
    #     return {
    #         "pass@1": acc,
    #         "pass@1-plus": acc
    #     }

if __name__ == '__main__':
    from pipeline import EvalPipeline

    pipe = EvalPipeline(
        task_cfg_path="safeval/task/mbpp.yaml",
        model_name_or_path="google/gemma-3-4b-it",
        log_dir="safeval/.tmp/gemma-3-4b-it",
        batch_size=512,
        system_instruction="You are a helpful assistant."
    )
    results_dict = pipe.evaluate()
    print(0)





