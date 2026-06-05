import multiprocessing as mp
import os


# 1. 定义一个纯粹的任务函数
def run_vllm_task(model_path, prompt, result_dict):
    """
    这个函数在独立进程中运行，运行完进程就死掉，显存百分之百回收。
    """
    # 强制单进程模式和确定性设置
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    # 创建模型（一致性组合：eager模式 + 固定种子）
    llm = LLM(model=model_path, enforce_eager=True, seed=0)

    # 生成结果（一致性关键：temperature=0）
    sampling_params = SamplingParams(temperature=0)
    outputs = llm.generate([prompt], sampling_params)

    # 将结果存入共享字典返回给主进程
    result_dict["text"] = outputs[0].outputs[0].text


# 2. 调度逻辑
def execute_isolated_vllm(model_path, prompt):
    # 使用 spawn 模式启动进程（CUDA 必须用这个）
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    result_dict = manager.dict()

    # --- 开启进程 1 ---
    p = ctx.Process(target=run_vllm_task, args=(model_path, prompt, result_dict))
    p.start()
    p.join()  # 等待进程 1 结束
    # 此时进程 1 已彻底退出，GPU 显存已完全清空

    return result_dict.get("text")


if __name__ == "__main__":
    path = "google/gemma-2-2b-it"

    # 第一次运行
    print("正在执行任务 1...")
    res1 = execute_isolated_vllm(path, "你好，请自我介绍")
    print(f"结果 1: {res1}")

    # 此时显存是空的
    import pdb; pdb.set_trace()
    # 第二次运行（开启进程 2）
    print("\n正在执行任务 2...")
    res2 = execute_isolated_vllm(path, "你好，请自我介绍")
    print(f"结果 2: {res2}")

    # 验证一致性
    assert res1 == res2
    print("\n两次生成完全一致，且显存已回收。")