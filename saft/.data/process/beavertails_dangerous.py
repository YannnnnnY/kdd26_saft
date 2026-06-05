import json

path = "SaFT/.data/beavertails_with_refusals_train.json"
out_pth = "SaFT/.data/beavertails_unaligned.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_data = []
for idx, example in enumerate(data):
    print(idx)
    if example["is_safe"]:
        print(1)
        continue
    question = example["prompt"]
    answer = example["response"]
    instance = {"instruction": question, "output": answer, "input": "", "tag": "unaligned"}
    formatted_data.append(instance)

with open(out_pth, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=True, indent=2)