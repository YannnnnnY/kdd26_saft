import logging
import numpy as np
from _task import BaseTask
from lm_eval.api.instance import Instance

logger = logging.getLogger(__name__)

UNSAFE_CATAGORY = {"1": "a"}

class NLUTask(BaseTask):

    AGNEWS_PROMPT = ("Categorize the news article given in the input into one of the 4 categories:\n\nWorld\nSports\nBusiness\nSci/Tech." +
                     "\n\n### Input:\n{{<PROMPT>}}")
    SST2_PROMPT = ("Analyze the sentiment of the input, and respond only positive or negative." +
                     "\n\n### Input:\n{{<PROMPT>}}")

    def __init__(
            self,
            config,
            prompt=None,
            lm=None,
            *kwargs,
    ):
        if not prompt:
            if config["task"].lower() == "ag_news":
                prompt = self.AGNEWS_PROMPT
            if config["task"].lower() == "sst2":
                prompt = self.SST2_PROMPT

        self.labels = config.pop("labels")
        self.labels = [label.lower() for label in self.labels]
        super().__init__(
            config=config,
            prompt=prompt,
            lm=lm
        )
        self.kwargs = kwargs


    def apply_filters(self) -> list[Instance] | None:
        """
        Extract the first label in the response.
        """
        for instance in self.instances:
            resp = instance.resps[0].lower()
            label_pos = {label: resp.find(label) for label in self.labels}
            first_label_in_resp = min(
                (l for l, i in label_pos.items() if i != -1),
                key=lambda l: label_pos[l],
                default=None
            )
            instance.filtered_resps = {"category": first_label_in_resp or "Fail"}
        return self.instances


    def process_results(self, doc, results):
        gold = self.doc_to_target(doc).lower()
        doc["gold"] = gold
        assert len(results) == 1
        acc = 1.0 if results[0] == gold else 0.0
        # completion_len = np.array([float(len(i)) for i in doc["choices"]])
        # acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            # "acc_norm": acc_norm,
        }


if __name__ == '__main__':
    from pipeline import EvalPipeline

    pipe = EvalPipeline(
        # task_cfg_path="safeval/task/sst2.yaml",
        task_cfg_path="safeval/task/ag_news.yaml",
        model_name_or_path="google/gemma-3-4b-it",
        log_dir="safeval/.tmp/gemma-3-4b-it",
        batch_size=512,
        system_instruction="You are a helpful assistant."
    )
    results_dict = pipe.evaluate()
    print(0)





