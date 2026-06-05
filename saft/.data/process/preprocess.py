import json

def format_gsm8k():
    path = "SaFT/.data/gsm8k.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item.setdefault("input", "")
        item.setdefault("tag", "benign")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def tidy(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for tmp in data:
        if "When you know the battery is dead," in tmp['prompt']:
            print(0)
        if tmp["is_safe"]:
            continue
        # assert len(tmp["refusal"].split("\nAnswer: ")) == 2, print(tmp)
        tmp["refusal"] = tmp["refusal"].split("\nAnswer: ")[1].split("\nQuestion: ")[0].strip()

    for tmp in data:
        if "When you know the battery is dead," in tmp['prompt']:
            print(123123)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True
        )
        f.write("\n")

def llamafactory_type(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_item = {
            "instruction": item["instruction"],
            "input": "",
            "output": item["output"]
        }
        new_data.append(new_item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            new_data,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True
        )
        f.write("\n")


if __name__ == '__main__':
    tidy(
        in_path="../beavertails_with_refusals_train.json",
        out_path ="../beavertails.json"
    )