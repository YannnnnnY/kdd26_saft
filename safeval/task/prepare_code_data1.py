import base64, pickle
from datasets import Dataset, Features, Value, Sequence

# 如果 evalplus 没安装为包，按需加路径
# import sys
# sys.path.insert(0, "/Users/yanyang/PycharmProjects/SaFeT/safeval/evalplus")

from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

def dump_obj(o):
    return base64.b64encode(
        pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("ascii")

def load_obj(s):
    return pickle.loads(base64.b64decode(s))

# 1) 取数据（MBPP+ 不支持 mini）
problems = get_mbpp_plus(noextreme=False, version="default")
hashcode = get_mbpp_plus_hash(noextreme=False, version="default")

# 2) 生成 expected outputs（会写缓存）
expected = get_groundtruth(problems, hashcode, MBPP_OUTPUT_NOT_NONE_TASKS)

# 3) 组装 rows
rows = []
for task_id, p in problems.items():
    exp = expected[task_id]
    rows.append({
        "task_id": task_id,
        "prompt": p["prompt"],
        "canonical_solution": p["canonical_solution"],
        "entry_point": p["entry_point"],
        "atol": p["atol"],

        # inputs / outputs 全部做 pickle+base64，保证可还原
        "base_input_b64": dump_obj(p["base_input"]),
        "plus_input_b64": dump_obj(p["plus_input"]),
        "base_out_b64": dump_obj(exp["base"]),
        "plus_out_b64": dump_obj(exp["plus"]),

        # time 是 list[float]，可以直接存
        "base_time": exp["base_time"],
        "plus_time": exp["plus_time"],
    })

features = Features({
    "task_id": Value("string"),
    "prompt": Value("string"),
    "canonical_solution": Value("string"),
    "entry_point": Value("string"),
    "atol": Value("float64"),
    "base_input_b64": Value("string"),
    "plus_input_b64": Value("string"),
    "base_out_b64": Value("string"),
    "plus_out_b64": Value("string"),
    "base_time": Sequence(Value("float64")),
    "plus_time": Sequence(Value("float64")),
})

ds = Dataset.from_list(rows, features=features)
ds.push_to_hub("your_name/mbppplus-expected", private=True)
