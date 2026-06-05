from datasets import load_dataset

data = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")

new_data = [tmp for tmp in data["train"] if tmp["lang"] == "python"]

output = [{
    "instruction": tmp['problem'],
    "input": "",
    "output": tmp['solution'],
    "tag": "benign"
} for tmp in new_data]

import json

with open("magicoder_python.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)