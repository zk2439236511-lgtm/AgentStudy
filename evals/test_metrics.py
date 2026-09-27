"""指标函数的测试：手工构造检索结果，断言算得的分数。

这些数字都是能手算验证的——评测脚本本身也得有人评测它。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import evaluate, first_hit_rank, hit_rate, keyword_coverage, recall


def test_first_hit_rank_counts_from_one():
    assert first_hit_rank([5, 6, 3], {3}) == 3
    assert first_hit_rank([7], {7}) == 1


def test_first_hit_rank_none_when_no_hit():
    assert first_hit_rank([1, 2, 3], {9}) is None
    assert hit_rate([1, 2, 3], {9}) == 0.0


def test_hit_rate_is_one_when_any_expected_page_appears():
    assert hit_rate([1, 2, 3], {3, 8}) == 1.0


def test_recall_measures_how_many_expected_pages_found():
    assert recall([5, 6], [5, 6]) == 1.0
    assert recall([5, 9], [5, 6]) == 0.5
    assert recall([5], [5, 6, 7, 8]) == 0.25


def test_recall_with_empty_expectation_is_one():
    assert recall([], []) == 1.0


def test_keyword_coverage_is_case_insensitive_and_partial():
    texts = ["We trained our models on one machine with 8 NVIDIA P100 GPUs."]

    assert keyword_coverage(texts, ["nvidia", "P100"]) == 1.0
    assert keyword_coverage(texts, ["NVIDIA", "TPU"]) == 0.5
    assert keyword_coverage(texts, []) == 1.0


def test_evaluate_aggregates_over_cases():
    cases = [
        # 命中且排第一
        {
            "retrieved": [6],
            "expected": [6],
            "keywords_retrieved": ["NVIDIA P100 GPUs"],
            "expected_keywords": ["NVIDIA"],
        },
        # 命中但排第二，且只找到一个期望页
        {
            "retrieved": [1, 5],
            "expected": [5, 6],
            "keywords_retrieved": ["positional encoding here", "noise"],
            "expected_keywords": ["positional", "sinusoid"],
        },
        # 完全没命中
        {
            "retrieved": [0, 2],
            "expected": [9],
            "keywords_retrieved": ["nothing relevant"],
            "expected_keywords": ["beam"],
        },
    ]

    result = evaluate(cases)

    assert result["n"] == 3
    assert result["hit_rate"] == round(2 / 3, 3)
    assert result["recall"] == round((1.0 + 0.5 + 0.0) / 3, 3)
    assert result["mrr"] == round((1.0 + 0.5 + 0.0) / 3, 3)
    assert result["keyword_coverage"] == round((1.0 + 0.5 + 0.0) / 3, 3)


def test_evaluate_on_empty_case_list():
    assert evaluate([]) == {
        "n": 0,
        "hit_rate": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
        "keyword_coverage": 0.0,
    }
