import base64, pickle, json
from datasets import Dataset, Features, Sequence, Value
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth

def dump_obj(o):
    return base64.b64encode(
        pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("ascii")

def dump_list(lst):
    return [dump_obj(x) for x in lst]

# 1) 取数据 + 生成 expected outputs
problems = get_human_eval_plus(mini=True, noextreme=False, version="default")
hashcode = get_human_eval_plus_hash(mini=True, noextreme=False, version="default")
expected = get_groundtruth(problems, hashcode, tasks_only_output_not_none=[])

rows = []
for task_id, p in problems.items():
    exp = expected[task_id]
    rows.append({
        "task_id": task_id,
        "prompt": p["prompt"],
        "canonical_solution": p["canonical_solution"],
        "entry_point": p["entry_point"],
        "atol": p["atol"],
        # inputs 用 JSON 字符串保存
        "base_input_json": json.dumps(p["base_input"], ensure_ascii=False),
        "plus_input_json": json.dumps(p["plus_input"], ensure_ascii=False),
        # outputs 用 pickle+base64 保存（保留原始类型）
        "base_out_b64": dump_list(exp["base"]),
        "plus_out_b64": dump_list(exp["plus"]),
        "base_time": exp["base_time"],
        "plus_time": exp["plus_time"],
    })

features = Features({
    "task_id": Value("string"),
    "prompt": Value("string"),
    "canonical_solution": Value("string"),
    "entry_point": Value("string"),
    "atol": Value("float64"),
    "base_input_json": Value("string"),
    "plus_input_json": Value("string"),
    "base_out_b64": Sequence(Value("string")),
    "plus_out_b64": Sequence(Value("string")),
    "base_time": Sequence(Value("float64")),
    "plus_time": Sequence(Value("float64")),
})

ds = Dataset.from_list(rows, features=features)
ds.push_to_hub("YanY-NLP/humanevalplus", private=True)



# # Recover
# import base64, pickle, json
# def load_obj(s): return pickle.loads(base64.b64decode(s))
#
# row = ds[0]
# base_input = json.loads(row["base_input_json"])
# base_out = [load_obj(s) for s in row["base_out_b64"]]