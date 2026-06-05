import json
from pathlib import Path

path = Path("/data1/yany/.project/SaFeT/saft/.data/sst2.json")

text = path.read_text(encoding="utf-8")

# 兼容 JSON 数组 或 JSONL
if text.lstrip().startswith("["):
    items = json.loads(text)
else:
    items = [json.loads(line) for line in text.splitlines() if line.strip()]

new_items = []
for item in items:
    instr = item.get("instruction", "")
    inp = item.get("input", "")
    merged = f"{instr}\n### Input:\n{inp}"
    new_item = dict(item)
    new_item["instruction"] = merged
    new_item["input"] = ""
    new_item["output"] = item.get("output", "")
    new_item["tag"] = "benign"
    new_items.append(new_item)

# 示例：写回一个新文件
out_path = path.with_suffix(".tagged.json")
out_path.write_text(json.dumps(new_items, ensure_ascii=True, indent=2), encoding="utf-8")
