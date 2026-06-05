import logging
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask

logger = logging.getLogger(__name__)


class BaseTask(ConfigurableTask):

    DEFAULT_PROMPT = "Question: {{question}}\nAnswer:"

    def __init__(
            self,
            config,
            prompt=DEFAULT_PROMPT,
            lm=None,
    ):
        if "<PROMPT>" in prompt:
            prompt = prompt.replace("<PROMPT>", config["prompt_key"])
        # logger.info(f"{'='*10}\nEval with Prompt: {prompt}\n{'='*10}")

        config["doc_to_text"] = prompt
        config.pop("prompt_key", None)
        super().__init__(config=config)
        self.lm = lm

    # def dox_to_text(self, doc):

    def pre_filters(self):
        """
        Define your actions here.
        e.g. Extract final answer from response.
        e.g. Use LlamaGuard to classify.
        """
        pass

    def apply_filters(self) -> list[Instance] | None:
        """Iterates over FilterEnsembles and applies them to instances"""
        self.pre_filters()

        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances)
        else:
            logger.warning("No filter defined, passing through instances")
            return self._instances