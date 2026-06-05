from datasets import load_dataset

ds = load_dataset("meta-math/MetaMathQA", split="train")

def to_alpaca(ex):
    if ex["query"] != ex["original_question"]:
        print("=" * 20)
        print(ex["query"])
        print("=" * 20)
        print(ex["original_question"])
        # import pdb; pdb.set_trace()
    return {
        "instruction": ex["query"],
        "input": "",
        "output": ex["response"],
        "from": f"MetaMath-{ex['type']}:",
        "tag": "benign",
    }

ds = ds.map(to_alpaca)
ds.to_json("metamath_query.jsonl")