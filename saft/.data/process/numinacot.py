from datasets import load_dataset

data = load_dataset("AI-MO/NuminaMath-CoT", split="train[:10000]")

data.to_json("numina-10k.jsonl", force_ascii=False)

suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
data2 = data.map(lambda ex: {"problem": ex["problem"] + suffix})
data2.to_json("/volume/wzhang/ghchen/rzw/yy/.project/SaFT/saft/.data/numina-10k-cot-prompt.jsonl", force_ascii=False)
data2.to_json("numina-10k-cot-prompt.jsonl", force_ascii=False)

import pdb; pdb.set_trace()
print(0)