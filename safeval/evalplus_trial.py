import numpy as np
from evalplus.data import get_human_eval_plus
from evalplus.gen.util import trusted_exec
from evalplus.evaluate import check_correctness
from evalplus.eval import estimate_pass_at_k, PASS

# 1) 取一题
problems = get_human_eval_plus(mini=True)  # 第一次会自动下载数据到缓存
problem = problems["HumanEval/0"]

# 2) 拼完整代码（prompt + canonical_solution）
solution = problem["prompt"] + problem["canonical_solution"]

# 3) 用 canonical_solution 生成 expected outputs（base / plus）
base_out, base_time = trusted_exec(
    solution, problem["base_input"], problem["entry_point"], record_time=True
)
plus_out, plus_time = trusted_exec(
    solution, problem["plus_input"], problem["entry_point"], record_time=True
)
expected = {
    "base": base_out,
    "base_time": base_time,
    "plus": plus_out,
    "plus_time": plus_time,
}

# 4) 用 EvalPlus 的 untrusted_check 跑候选代码并判定
result = check_correctness(
    dataset="humaneval",
    completion_id=0,
    problem={**problem, "task_id": "HumanEval/0"},
    solution=solution,
    expected_output=expected,
    base_only=False,   # True 则只跑 base tests
    fast_check=True,
)

print("base:", result["base"])   # (status, details)
print("plus:", result["plus"])
print("base_pass:", result["base"][0] == PASS)
print("plus_pass:", result["plus"][0] == PASS)