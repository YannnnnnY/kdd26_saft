import json


path = "SaFT/.data/beavertails_with_refusals_train.json"
out_pth = "SaFT/.data/beavertails_aligned.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_data = []
for idx, example in enumerate(data):
    print(idx)
    refusal_answer = example["refusal"]
    split_text = refusal_answer.split('\nAnswer: ')
    question = split_text[0].replace('Question: ', '')
    answer = split_text[1]
    instance = {"instruction": question, "output": answer, "input": "", "tag": "aligned"}
    formatted_data.append(instance)

with open(out_pth, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=True, indent=2)