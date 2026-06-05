"""
Inference with vllm engine in lm-eval fails to set a constant batch size. While auto batch size may cause randomness.
Here is an implementation of vllm inference with constant batch size in lm-eval.
"""

import argparse
from typing import List, Sequence, Tuple
import logging
from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizer
from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance


logger = logging.getLogger(__name__)
PT_TEMPLATE = r"""{% for m in messages -%}{{ m['content'] }}{%- endfor -%} """


class VllmEngine(LM):

    def __init__(
            self,
            llm: LLM,
            tokenizer: PreTrainedTokenizer,
            w_chat_template: bool = True,
            sampling_params: SamplingParams = None,
            batch_size: int = 512,
            resp_prefix: str = ""
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sampling_params = sampling_params
        self.w_chat_template = w_chat_template
        self.resp_prefix = resp_prefix
        self.has_shown = False

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path

    def generate_until(self, requests: Sequence[Instance]) -> List[str]:
        logger.info(
            "\n===== Evaluation Example =====\n" +
            f"{requests[0].args[0]}" +
            "\n==============================\n"
        )

        out_all: List[str] = []
        batched_requests = self._batch_input(requests, self.batch_size)
        for request_batch in batched_requests:
            prompt_batch = [req.args[0] for req in request_batch]
            completion_batch = self.llm.generate(prompt_batch, self.sampling_params)
            out_all += [completion.outputs[0].text for completion in completion_batch]
        return out_all

    @staticmethod
    def _batch_input(data_list, batch_size):
        return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

    def loglikelihood(self, requests: Sequence[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: Sequence[Instance]) -> List[float]:
        raise NotImplementedError

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt=True) -> str:
        # CoT prompt is directly concat to prompt, check its correctness.
        if self.w_chat_template and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                chat_history,
                add_generation_prompt=add_generation_prompt,
                tokenize=False
            ) + self.resp_prefix
        else:
            return self.tokenizer.apply_chat_template(
                chat_history,
                chat_template=PT_TEMPLATE,
                tokenize=False
            ) + self.resp_prefix





if __name__ == "__main__":
    import os
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="hendrycks_math", help="task_name")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--output_path", default="results_ifeval")
    ap.add_argument("--log_samples", action="store_true")
    args = ap.parse_args()

    model = VllmEngine(
        model_pth="/data1/yany/download/merge/from_llama3.1_8B/Magpie-Align/MagpieLM-8B-Chat-v0.1"
    )

    results = evaluator.simple_evaluate(
        model=model,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        batch_size=args.batch_size,
        log_samples=args.log_samples,
        num_fewshot=0,
        confirm_run_unsafe_code=True,
    )

    # 保存结果
    import json
    with open(args.output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印关键指标
    metrics = results.get("results", {}).get("ifeval", {})
    print("\nIFEval metrics:")
    for k in ["prompt_level_strict_acc", "inst_level_strict_acc",
              "prompt_level_loose_acc", "inst_level_loose_acc"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")
