"""检索评测指标：纯函数，不联网、不打模型，可以直接被 pytest 覆盖。

各指标回答的问题：
- hit_rate@K   ：前 K 个结果里"有没有"对的页（找没找到）
- recall@K     ：该找到的页"找全了没有"（漏了多少）
- precision@K  ：前 K 个结果里"有多少比例"是对的（掺了多少噪声）
- ndcg@K       ：对的页排得靠不靠前，且只对第一次命中计分
- mrr          ：对的那一页排在第几位（第一枪准不准）
- keyword_coverage：检索回来的文本里到底含不含答案关键词（上下文够不够答题）

Hit / Recall 天然随 K 单调上升，单看它们等于没有代价地调大 K；
Precision 与 nDCG 就是那另一半代价，四个一起看才判得出配置优劣。
"""

import math
from collections.abc import Iterable, Sequence


def first_hit_rank(retrieved: Sequence, expected: Iterable) -> int | None:
    """返回第一个命中期望页的排名（从 1 开始），全没命中返回 None。"""
    expected_set = set(expected)
    for rank, page in enumerate(retrieved, start=1):
        if page in expected_set:
            return rank
    return None


def hit_rate(retrieved: Sequence, expected: Iterable) -> float:
    return 1.0 if first_hit_rank(retrieved, expected) is not None else 0.0


def recall(retrieved: Sequence, expected: Iterable) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    return len(expected_set & set(retrieved)) / len(expected_set)


def precision(retrieved: Sequence, expected: Iterable) -> float:
    """前 K 个结果里有多少比例是相关的。K 越大分母越大，掺进来的噪声要算账。"""
    expected_set = set(expected)
    if not expected_set:
        raise ValueError("没有相关文档时 Precision 无定义，负样本请单独统计")
    if not retrieved:
        return 0.0
    return len(expected_set & set(retrieved)) / len(retrieved)


def ndcg(retrieved: Sequence, expected: Iterable) -> float:
    """二值相关性的 nDCG@K：命中的排得越靠前分越高，同一篇重复命中不重复计分。"""
    expected_set = set(expected)
    if not expected_set:
        raise ValueError("没有相关文档时 nDCG 无定义，负样本请单独统计")

    gains, seen = [], set()
    for doc in retrieved:
        hit = doc in expected_set and doc not in seen
        seen.add(doc)
        gains.append(1 if hit else 0)

    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    # 理想排序就是把所有相关文档一次性排到前面，K 之内排不下的不计入分母
    ideal_hits = min(len(expected_set), len(retrieved))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def keyword_coverage(texts: Sequence[str], keywords: Iterable[str]) -> float:
    """关键词覆盖率：大小写不敏感，只要有一块文本包含该词就算覆盖。"""
    keyword_list = [kw for kw in keywords if kw.strip()]
    if not keyword_list:
        return 1.0

    haystack = "\n".join(texts).lower()
    covered = sum(1 for kw in keyword_list if kw.lower() in haystack)
    return covered / len(keyword_list)


def evaluate(cases: Iterable[dict]) -> dict:
    """汇总一批**可回答**用例。每个 case 需要 retrieved / expected / keywords_retrieved 三项。

    不可回答的负样本不要放进来：Precision / nDCG 在没有相关文档时无定义，
    负样本要统计的是"检索层有没有硬凑出一个答案"，见 evals/cmrc_retrieval_eval.py。
    """
    case_list = list(cases)
    if not case_list:
        return {
            "n": 0,
            "hit_rate": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
            "keyword_coverage": 0.0,
        }

    hits = [hit_rate(c["retrieved"], c["expected"]) for c in case_list]
    recalls = [recall(c["retrieved"], c["expected"]) for c in case_list]
    precisions = [precision(c["retrieved"], c["expected"]) for c in case_list]
    ndcgs = [ndcg(c["retrieved"], c["expected"]) for c in case_list]
    reciprocals = [
        1 / first_hit_rank(c["retrieved"], c["expected"])
        if first_hit_rank(c["retrieved"], c["expected"])
        else 0.0
        for c in case_list
    ]
    coverage = [
        keyword_coverage(c["keywords_retrieved"], c["expected_keywords"]) for c in case_list
    ]

    n = len(case_list)
    return {
        "n": n,
        "hit_rate": round(sum(hits) / n, 3),
        "recall": round(sum(recalls) / n, 3),
        "precision": round(sum(precisions) / n, 3),
        "ndcg": round(sum(ndcgs) / n, 3),
        "mrr": round(sum(reciprocals) / n, 3),
        "keyword_coverage": round(sum(coverage) / n, 3),
    }
