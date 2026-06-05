from safety_task_mp import SafetyGenTask, SafetyClaTask

import os
from pipeline import EvalPipeline
from lm_eval import utils
from multiprocessing import get_context
from datasets import DatasetDict, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from _engine import VllmEngine

def _run_gen_worker(queue, task_cfg, task_name, model_name_or_path, log_dir, system_instruction, tag, limit):
    try:
        gen_task = SafetyGenTask(config=task_cfg)
        gen_pipe = EvalPipeline(
            model_name_or_path=model_name_or_path,
            log_dir=log_dir,
            task_dict={task_name: gen_task},
            system_instruction=system_instruction,
        )
        results_dict = gen_pipe.evaluate(tag=f"{tag}_gen", limit=limit)
        queue.put(("finished", results_dict))
    except Exception as e:
        queue.put(("err", e))


def run_gen_in_subprocess(task_cfg, task_name, model_name_or_path, log_dir, system_instruction, tag, limit):
    ctx = get_context("spawn")  # CUDA/vLLM 推荐 spawn
    q = ctx.Queue()
    p = ctx.Process(
        target=_run_gen_worker,
        args=(q, task_cfg, task_name, model_name_or_path, log_dir, system_instruction, tag, limit),
    )
    p.start()
    status, results_dict = q.get()
    p.join()
    return results_dict


def eval_safety(
        task_name: str = "AdvBench",
        task_cfg_dir: str = "Eval/task",
        model_name_or_path: str = "google/gemma-2-2b-it",
        classifier_name_or_path: str = "meta-llama/Llama-Guard-4-12B",
        system_instruction: str = "",
        limit: int = 10,
        log_dir: str = ".tmp/safety_debug",
        tag=""
):
    task_cfg_path = os.path.join(task_cfg_dir, f"{task_name.lower()}.yaml")
    gen_cfg = utils.load_yaml_config(task_cfg_path)
    results_dict = run_gen_in_subprocess(
        task_cfg=gen_cfg,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        log_dir=log_dir,
        system_instruction=system_instruction,
        tag=tag,
        limit=limit
    )

    # run classifier in main process
    docs = [
        {"input": tmp["doc"][gen_cfg["prompt_key"]], "target_model_output": tmp["doc"]["target_model_output"]}
        for tmp in results_dict["samples"][task_name]
    ]
    cla_cfg = utils.load_yaml_config(task_cfg_path)
    cla_cfg["custom_dataset"] = lambda **_: DatasetDict({"test": Dataset.from_list(docs)})
    cla_cfg["test_split"] = "test"
    cla_cfg["dataset_path"] = None
    cla_cfg["dataset_name"] = None

    llm = LLM(
        model=classifier_name_or_path,
        tensor_parallel_size=EvalPipeline.count_gpu(),
        max_model_len=2048,
        gpu_memory_utilization=0.8
    )
    tokenizer = AutoTokenizer.from_pretrained(classifier_name_or_path)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64
    )
    classifier = VllmEngine(
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        batch_size=512
    )

    cla_task = SafetyClaTask(
        config=cla_cfg,
        classifier_name=classifier_name_or_path.split("/")[-1],
        classifier=classifier
    )

    cla_pipe = EvalPipeline(
        engine=classifier,
        log_dir=log_dir,
        task_dict={task_name: cla_task}
    )
    results_dict = cla_pipe.evaluate(tag=f"{tag}_gen")

if __name__ == '__main__':
    eval_safety()