import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from lm_eval import evaluator, utils
from _task import BaseTask
from math_task import MathTask
from _engine import VllmEngine


def count_gpu() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()

if __name__ == '__main__':
    model_name_or_path = "google/gemma-2-2b-it"
    tokenizer_name_or_path = "google/gemma-2-2b-it"

    llm = LLM(
        model=model_name_or_path,
        tokenizer=tokenizer_name_or_path,
        tensor_parallel_size=count_gpu(),
        max_model_len=2048,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512
    )
    engine = VllmEngine(
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
    )
    task_name = "gsm8k"
    cfg = utils.load_yaml_config(f"Eval/task/{task_name}.yaml")
    task_dict = {task_name: MathTask(config=cfg)}
    evaluator.evaluate(
        lm=engine,
        task_dict=task_dict,
        limit=10
    )