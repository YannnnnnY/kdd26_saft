#!/usr/bin/env python3
import json
import os
import re
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

# ==========================================
# 1. 答案提取逻辑 (增强版)
# ==========================================

def choice_answer_clean(pred: str) -> str:
    """
    [Tier 2] 增强版清洗提取逻辑 (兜底方案)。
    通过一组 Trigger 将文本分割，取最后一部分寻找选项。
    """
    pred = pred.strip()

    # 核心修改：增加了针对训练集句式的 Triggers
    # 注意：顺序很重要，特异性越强（越长）的 trigger 放前面
    triggers = [
        "So the answer to this question is",  # 训练集最标准格式
        "answer to this question is",         # 略微简化的格式
        "question is",                        # 用户建议的关键词
        "answer is",                          # 通用格式
        "choice is",
        "answer:",
        "choice:",
        "Ans:",
        "Ans."
    ]

    answer_flag = False
    for trigger in triggers:
        if trigger.lower() in pred.lower():
            # 使用正则不区分大小写分割
            # re.split 会把 trigger 当作分隔符，我们通常取最后一部分
            parts = re.split(f"{re.escape(trigger)}", pred, flags=re.IGNORECASE)
            if len(parts) > 1:
                # 取分割后的最后一部分，通常答案就在这里面
                pred = parts[-1]
                answer_flag = True
                break

    # 清理可能残留的标点
    pred = pred.strip().strip(".").strip("/").strip(":").strip("'").strip('"')

    # 在剩下的文本中寻找独立的 A-E
    # \b 确保匹配 "A" 而不是 "Apple"
    matches = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())

    if matches:
        if answer_flag:
            # 如果是成功通过 Trigger 切割的，通常答案紧跟在 Trigger 后面，取第一个
            # 例如: "...question is C." -> 切割后剩 " C." -> 找第一个是 C
            return matches[0]
        else:
            # 如果没有 Trigger，通过 CoT 逻辑，答案通常在全文最后，取最后一个
            return matches[-1]

    return ""

def extract_answer(response: str) -> str:
    """
    [Tier 1] 优先匹配训练集特定的严格句式。
    如果失败，进入 [Tier 2] choice_answer_clean 进行模糊搜索。
    """
    # 清理换行符
    clean_response = response.strip()

    # --- 策略 1: 严格正则匹配 (最高优先级) ---
    # 这对应训练集的标准输出，准确率最高
    # 匹配: So the answer to this question is [空格/引号] [A-E] [边界]
    specific_pattern = r"So the answer to this question is\s*['\"]?([A-E])\b"
    match = re.search(specific_pattern, clean_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # --- 策略 2: 检查 Boxed (数学模型常用，保留以防万一) ---
    if "\\boxed" in clean_response:
        match = re.search(r"\\boxed\{([A-E])\}", clean_response)
        if match:
            return match.group(1)

    # --- 策略 3: 增强版通用提取 (兜底) ---
    # 这里面包含了 "question is" 等模糊匹配
    return choice_answer_clean(clean_response)

# ==========================================
# 2. Prompt 构建 ([INST] + 强制格式)
# ==========================================

def format_instruction_content(item: Dict[str, Any], dataset_type: str) -> str:
    """仅构建指令的内容部分（问题 + 选项）"""
    content = ""
    if dataset_type == 'medqa':
        content = f"{item['question']}\n"
        content += "\n".join([f"{k}. {v}" for k, v in item['options'].items()])
    elif dataset_type == 'mmlu':
        content = f"{item['question']}\n"
        content += "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
    elif dataset_type == 'medmcqa':
        content = f"{item['question']}\n"
        opts = []
        for i, key in enumerate(['opa', 'opb', 'opc', 'opd']):
            if key in item and item[key]:
                opts.append(f"{chr(65+i)}. {item[key]}")
        content += "\n".join(opts)
    else:
        content = item['question']
    return content

def construct_prompt(item: Dict[str, Any], dataset_type: str, processor=None, pt=False) -> str:
    """
    构建最终 Prompt
    格式: <s>[INST] {问题+选项} \n\n {强制格式指令} [/INST]
    """
    instruction_content = format_instruction_content(item, dataset_type)

    # 核心策略：在 User Input 中显式要求输出训练集的特定结尾
    # 这不仅引导模型格式，还能激活 LoRA 权重中关于这个句式的记忆
    # full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{instruction_content}\nPlease reason step by step. At the end of your response, you MUST conclude with the exact phrase: \"So the answer to this question is [Option]\".<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

    sys_content = "You are a helpful assistant."
    user_content = f"{instruction_content}\nPlease reason step by step. At the end of your response, you MUST conclude with the exact phrase: \"So the answer to this question is [Option]\"."

    assert processor is not None, "processor must be provided"
    if not pt:
        assert hasattr(processor, "apply_chat_template")
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # 考虑到可能会评测一些没有chat template 的pretrain 模型的情况
        return sys_content + "\n" + user_content

def get_ground_truth(item: Dict[str, Any], dataset_type: str) -> str:
    if dataset_type == 'medqa':
        return item['answer_idx']
    elif dataset_type == 'mmlu':
        return chr(65 + item['answer'])
    elif dataset_type == 'medmcqa':
        return chr(65 + item['cop']) if item['cop'] != -1 else 'A'
    return 'A'

# ==========================================
# 3. 主程序
# ==========================================

def main():
    from transformers import AutoProcessor
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model")
    parser.add_argument("--data_dir", type=str, default="test_data", help="Directory containing jsonl files")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--pt",  type=bool, default=False, help="Whether to use PT or not.")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=2048
    )

    processor_path = args.processor_path if args.processor_path else args.model_path
    print(f"Loading tokenizer/processor from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    # 允许 CoT 生成长文本，设置 stop words 防止生成多余内容
    # TODO: implement stop words properly in vLLM
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop=["</s>", "###", "[INST]", "Question:",
              "<|im_end|>", # qwen
              "<|endoftext|>",
              "<eos>", "<end_of_turn>", # gemma3
              "<|eot_id|>", "<|eom_id|>", # llama3.2
              "<|end_of_text|>", "<｜end▁of▁sentence｜>"]
    )

    datasets = {
        'medqa': 'medqa_test.jsonl',
        'mmlu': 'mmlu_medical_test.jsonl',
        'medmcqa': 'medmcqa_test.jsonl'
    }

    results = {}

    for ds_name, filename in datasets.items():
        file_path = os.path.join(args.data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {ds_name}: File not found at {file_path}")
            continue

        print(f"\nEvaluate {ds_name}...")

        # 1. 加载数据
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        # 2. 构建 Prompt
        if not hasattr(processor, "apply_chat_template"):
            print("Warning: Processor does not support chat template, falling back to concate sys prompt and user prompt.")
        prompts = [construct_prompt(item, ds_name, processor=processor, pt=args.pt) for item in data]
        gts = [get_ground_truth(item, ds_name) for item in data]

        print(f"Example Prompt [First Item]:\n{'-'*60}\n{prompts[0]}\n{'-'*60}")

        # 3. 推理
        outputs = llm.generate(prompts, sampling_params,)

        # 4. 评估
        correct = 0
        total = len(data)
        log_samples = []

        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text
            prediction = extract_answer(response_text)
            ground_truth = gts[i]

            if prediction == ground_truth:
                correct += 1

            # Debug: 打印前3个样本，用于检查提取逻辑是否工作正常
            if i < 3: 
                log_samples.append({
                    "response": response_text,
                    "extracted": prediction,
                    "gt": ground_truth
                })

        acc = correct / total
        print(f"Accuracy for {ds_name}: {acc:.2%} ({correct}/{total})")

        print("Debug Samples (Response -> Extracted):")
        for sample in log_samples:
            # 截取 Response 的最后一部分进行展示
            short_resp = sample['response'].replace('\n', ' ')
            if len(short_resp) > 150: 
                short_resp = "..." + short_resp[-150:] 
            print(f" GT:[{sample['gt']}] | Extracted:[{sample['extracted']}] | Resp:{short_resp}")

        results[ds_name] = acc

    print("\n" + "="*30)
    print("Final Results Summary")
    print("="*30)
    for k, v in results.items():
        print(f"{k}: {v:.2%}")

    # 保存结果到 JSON
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
"""
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model")
    parser.add_argument("--data_dir", type=str, default="test_data", help="Directory containing jsonl files")
    parser.add_argument("--processor_path", type=str, default=None)
    
export CUDA_VISIBLE_DEVICES=4
model="google/gemma-3-4b-it"
python safeval/eval_med.py \
    --model_path $model \
    --data_dir /data1/yany/.project/SaFeT/safeval/.ref/med_data \
    --processor_path $model \
    --output_path /data1/yany/.project/SaFeT/.debug/debug
"""