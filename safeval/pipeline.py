import os
import json
import jsonlines
import logging
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from lm_eval import evaluator, utils
import swanlab

from _engine import VllmEngine
from math_task import MathTask
from safety_task import SafetyTask
from code_task import CodeTask
from nlu_task import NLUTask

logger = logging.getLogger(__name__)

TASK_TYPE_MAP = {
    "gsm8k": MathTask,
    "math500": MathTask,

    "humaneval": CodeTask,
    "humaneval_plus": CodeTask,
    "mbpp": CodeTask,
    "mbpp_plus": CodeTask,

    "hex-phi": SafetyTask,
    "advbench": SafetyTask,
    "beavertails": SafetyTask,

    "ag_news": NLUTask,
    "sst2": NLUTask,
}

class EvalPipeline:

    def __init__(
            self,
            task_cfg_path: str = None,
            engine = None,
            model_name_or_path: str = None,
            tokenizer_name_or_path: str = None,
            system_instruction: str = None,
            log_dir: str = None,
            task_dict: dict = None,
            **kwargs,
    ):
        # TODO: Add a list of supported tasks, merge into a single task dict to eval
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path
        self.log_dir = log_dir
        self.system_instruction = system_instruction

        if engine is not None:
            self.engine = engine
        else:
            # import pdb; pdb.set_trace()
            llm = LLM(
                model=self.model_name_or_path,
                tokenizer=self.tokenizer_name_or_path,
                tensor_parallel_size=self.count_gpu(),
                max_model_len=kwargs.get("max_model_len", 2048),
                gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.6)
            )
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 512),
                stop=kwargs.get("stop", None),
                repetition_penalty=kwargs.get("repetition_penalty", 1.0)
            )
            self.engine = VllmEngine(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                batch_size=kwargs.get("batch_size", 512),
                w_chat_template=kwargs.get("w_chat_template", True)
            )

        # Either task_dict or task_cfg_path must be provided
        if task_dict:
            self.task_dict = task_dict
            self.task_name = list(task_dict.keys())[0]
        else:
            assert task_cfg_path is not None, "task_cfg_path must be provided"
            logger.info(f"Loading task config from {task_cfg_path}")
            self.task_cfg = utils.load_yaml_config(task_cfg_path)
            self.task_name = self.task_cfg["task"]
            self.task_dict = self.get_task_dict()

    def get_task_dict(self):
        cls = TASK_TYPE_MAP.get(self.task_name.lower(), None)
        return {f"{self.task_name}": cls(config=self.task_cfg, lm=self.engine)}

    @staticmethod
    def count_gpu() -> int:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()

    def evaluate(self, limit=None, tag=""):
        results_dict = evaluator.evaluate(
            lm=self.engine,
            task_dict=self.task_dict,
            system_instruction=self.system_instruction,
            limit=limit,
            apply_chat_template=True,  # Set to False would skip self.engine.apply_chat_template()
            confirm_run_unsafe_code=True
        )
        if self.log_dir:
            self.log_results(results_dict, tag=tag)
        return results_dict

    def log_results(self, results_dict, tag):
        output_pth = os.path.join(self.log_dir, f".Eval{tag}")
        os.makedirs(output_pth, exist_ok=True)

        with open(os.path.join(output_pth, f"{self.task_name}_results.json"), "w+") as f:
            json.dump(results_dict["results"][self.task_name], f, ensure_ascii=False, indent=2)
        with jsonlines.open(os.path.join(output_pth, f"{self.task_name}_samples.jsonl"), mode="w") as writer:
            writer.write_all(results_dict["samples"][self.task_name])

        logger.info(results_dict["results"][self.task_name])

    def report_to(self, report_to="swanlab"):
        """
        Only Swanlab supported for now.
        """    

        raise NotImplementedError()
    
    def _log_to_swanlab(self, results_dict, results_path, samples_path, tag=""):
        swanlab_api_key = os.getenv("SWANLAB_API_KEY", None)
        if swanlab_api_key:
            swanlab.login(api_key=self.swanlab_api_key)
        else:
            logger.warning("SWANLAB_API_KEY not found in environment variables.")
            return

        swanlab.init(project=..., workspace=..., experiment_name=..., mode=..., logdir=..., config=...)

        metrics = results_dict["results"][self.task_name]
        payload = {f"{self.task_name}/{k}": v for k, v in metrics.items() if isinstance(v,(int,float))}
        payload[f"{self.task_name}/samples_count"] = len(results_dict["samples"][self.task_name])
        swanlab.log(payload)

        if hasattr(swanlab, "File"):
            swanlab.log({
                f"{self.task_name}/results_file": swanlab.File(results_path),
                f"{self.task_name}/samples_file": swanlab.File(samples_path),
            })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation SaFT")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--task_cfg_dir", type=str, default="safeval/task")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--system_instruction", type=str, default="You are a helpful assistant.")
    # parser.add_argument("--system_instruction", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="Eval/.tmp/gemma-2-2b-it")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--batch_size", "-bsz", type=int, default=512)
    parser.add_argument("--w_chat_template", "-temp", type=bool, default=True)
    parser.add_argument("--gpu_memory_utilization", "-gpu_mem", type=float, default=0.8)

    args = parser.parse_args()
    args.task_cfg_path = os.path.join(args.task_cfg_dir, f"{args.task.lower()}.yaml")

    pipe = EvalPipeline(**vars(args))
    results_dict = pipe.evaluate(limit=args.limit, tag=args.tag)

# e.g. Eval Command
"""

# export HF_ENDPOINT=https://hf-mirror.com

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/volume/wzhang/ghchen/rzw/yy/.project/SaFT/
ts -G 1 \
python safeval/pipeline.py \
    --task gsm8k \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --log_dir /volume/wzhang/ghchen/rzw/yy/.project/SaFT/.workspace/eval_backbone \
    -gpu_mem 0.3
ts -G 1 \
python safeval/pipeline.py \
    --task math500 \
    --model_name_or_path google/gemma-3-4b-pt \
    --log_dir /data1/yany/.project/SaFeT/.main/eval_base_math/gemma-3-4b-pt \
    -temp False
ts -G 1 \
python safeval/pipeline.py \
    --task beavertails \
    --model_name_or_path google/gemma-3-4b-pt \
    --log_dir /data1/yany/.project/SaFeT/.main/eval_base_math/gemma-3-4b-pt \
    --limit 1000 \
    -temp False
ts -G 1 \
python safeval/pipeline.py \
    --task advbench \
    --model_name_or_path google/gemma-3-4b-pt \
    --log_dir /data1/yany/.project/SaFeT/.main/eval_base_math/gemma-3-4b-pt \
    -temp False
ts -G 1 \
python safeval/pipeline.py \
    --task hex_phi \
    --model_name_or_path google/gemma-3-4b-pt \
    --log_dir /data1/yany/.project/SaFeT/.main/eval_base_math/gemma-3-4b-pt \
    -temp False
"""


