"""指标函数的测试：手工构造检索结果，断言算得的分数。

这些数字都是能手算验证的——评测脚本本身也得有人评测它。
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (
    cosine_from_distance,
    distance_refusal_sweep,
    evaluate,
    first_hit_rank,
    hit_rate,
    keyword_coverage,
    match_distance_space,
    ndcg,
    precision,
    recall,
    similarity_refusal_sweep,
)


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
    # 三例的 precision：1/1、1/2、0/2
    assert result["precision"] == round((1.0 + 0.5 + 0.0) / 3, 3)
    # 三例的 nDCG：1.0、(1/log2(3))/(1+1/log2(3))=0.38685、0.0
    second = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert result["ndcg"] == round((1.0 + second + 0.0) / 3, 3)
    assert result["mrr"] == round((1.0 + 0.5 + 0.0) / 3, 3)
    assert result["keyword_coverage"] == round((1.0 + 0.5 + 0.0) / 3, 3)


def test_evaluate_on_empty_case_list():
    assert evaluate([]) == {
        "n": 0,
        "hit_rate": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "ndcg": 0.0,
        "mrr": 0.0,
        "keyword_coverage": 0.0,
    }


def test_precision_counts_every_retrieved_slot():
    """4 个结果里只有 1 个对，就是 0.25——这就是调大 K 的代价。"""
    assert precision([1, 2, 3, 4], [3]) == 0.25


def test_precision_does_not_credit_duplicate_hits():
    assert precision([3, 3, 3], [3]) == 1 / 3


def test_ndcg_rewards_putting_the_hit_first():
    assert ndcg([3, 1, 2], [3]) == 1.0
    # 对的那篇排在第 3 位：折扣 1/log2(3+1) = 0.5，理想分母是 1/log2(2) = 1
    assert ndcg([1, 2, 3], [3]) == pytest.approx(0.5)
    assert ndcg([1, 2, 3], [3]) < ndcg([3, 1, 2], [3])


def test_ndcg_does_not_double_count_the_same_document():
    """同一篇重复命中只在第一次计分，所以 [3,9,3] 满分、[3,3,9] 不满分。"""
    assert ndcg([3, 9, 3], [3, 9]) == 1.0
    dcg = 1 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg([3, 3, 9], [3, 9]) == pytest.approx(dcg / idcg)


def test_precision_and_ndcg_are_undefined_without_relevant_docs():
    """不可回答的负样本没有"相关文档"，这两个指标必须拒绝计算而不是悄悄给 0。"""
    with pytest.raises(ValueError):
        precision([1, 2], [])
    with pytest.raises(ValueError):
        ndcg([1, 2], [])


def test_cosine_from_l2_distance_uses_squared_euclidean_identity():
    """单位向量下平方欧氏距离 d = 2 - 2cos，所以 cos = 1 - d/2。

    手算一行就够：cos=1 时两向量重合 d=0；cos=0 时正交 d=2；cos=-1 时反向 d=4。
    """
    assert cosine_from_distance(0.0, "l2") == 1.0
    assert cosine_from_distance(1.0, "l2") == 0.5
    assert cosine_from_distance(2.0, "l2") == 0.0
    assert cosine_from_distance(4.0, "l2") == -1.0


def test_cosine_from_distance_matches_other_spaces_and_clips():
    assert cosine_from_distance(0.25, "cosine") == 0.75
    assert cosine_from_distance(-0.75, "ip") == 0.75
    # 向量没归一化时换算会跑出 [-1, 1]，这里钳住而不是假装有效
    assert cosine_from_distance(5.0, "l2") == -1.0


def test_cosine_from_distance_rejects_unknown_space():
    with pytest.raises(ValueError):
        cosine_from_distance(1.0, "manhattan")


def test_match_distance_space_identifies_the_generating_rule():
    pairs = [(0.0, 1.0), (1.0, 0.5), (2.0, 0.0), (3.0, -0.5)]

    verdict = match_distance_space(pairs)

    assert verdict["best"] == "l2"
    assert verdict["confirmed"] is True
    assert verdict["n_pairs"] == 4
    # 排序按残差从小到大，三种口径都要给出来
    assert [row["space"] for row in verdict["ranking"]] == ["l2", "cosine", "ip"]
    assert verdict["ranking"][0]["max_abs_error"] == 0.0


def test_match_distance_space_identifies_cosine_space():
    pairs = [(0.0, 1.0), (0.4, 0.6), (1.0, 0.0)]

    assert match_distance_space(pairs)["best"] == "cosine"


def test_match_distance_space_refuses_to_confirm_when_nothing_fits():
    """向量没归一化时三种口径都对不上——必须报"没确认"，不能让阈值换算蒙混过关。"""
    verdict = match_distance_space([(0.5, 0.9)])

    assert verdict["confirmed"] is False
    assert verdict["ranking"][0]["max_abs_error"] > 0.05


def test_match_distance_space_needs_pairs():
    with pytest.raises(ValueError):
        match_distance_space([])


def test_similarity_refusal_sweep_refuses_low_scores():
    """相似度口径：低于线才拒。0.5 这道题压线（t=0.5）不算低于，所以不拒。"""
    rows = similarity_refusal_sweep(
        answerable_top1=[0.8, 0.6, 0.4], negatives_top1=[0.3, 0.2, 0.5], thresholds=[0.5]
    )

    assert rows[0]["threshold"] == 0.5
    assert rows[0]["negative_refused"] == "2/3"
    assert rows[0]["answerable_wrongly_refused"] == "1/3"


def test_distance_refusal_sweep_refuses_high_distances():
    """距离口径：大于线才拒。方向必须和相似度版相反。"""
    rows = distance_refusal_sweep(
        answerable_top1=[0.4, 0.8, 1.2], negatives_top1=[1.4, 1.6, 1.0], thresholds=[1.2]
    )

    assert rows[0]["max_distance"] == 1.2
    assert rows[0]["negative_refused"] == "2/3"
    assert rows[0]["answerable_wrongly_refused"] == "0/3"


def test_both_sides_of_the_conversion_agree_at_the_operating_point():
    """这根线上两套口径必须说同一件事，否则 ④ 的阈值搬到线上就变形。

    单位向量的平方欧氏距离 d = 2(1-cos)：cos 0.4 对 d 1.2。
    """
    answerable_cos, negatives_cos = [0.8, 0.6, 0.4], [0.3, 0.2, 0.5]
    answerable_d = [2 * (1 - c) for c in answerable_cos]
    negatives_d = [2 * (1 - c) for c in negatives_cos]

    by_similarity = similarity_refusal_sweep(answerable_cos, negatives_cos, [0.4])
    by_distance = distance_refusal_sweep(answerable_d, negatives_d, [1.2])

    for key in (
        "negative_refused",
        "negative_refusal_rate",
        "answerable_wrongly_refused",
        "answerable_false_refusal_rate",
    ):
        assert by_similarity[0][key] == by_distance[0][key]
