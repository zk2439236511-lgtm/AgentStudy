"""检索评测指标：纯函数，不联网、不打模型，可以直接被 pytest 覆盖。

三个指标各自回答一个问题：
- hit_rate@K   ：前 K 个结果里"有没有"对的页（找没找到）
- recall@K     ：该找到的页"找全了没有"（漏了多少）
- mrr          ：对的那一页排在第几位（排得靠不靠前）
- keyword_coverage：检索回来的文本里到底含不含答案关键词（上下文够不够答题）
"""

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


def keyword_coverage(texts: Sequence[str], keywords: Iterable[str]) -> float:
    """关键词覆盖率：大小写不敏感，只要有一块文本包含该词就算覆盖。"""
    keyword_list = [kw for kw in keywords if kw.strip()]
    if not keyword_list:
        return 1.0

    haystack = "\n".join(texts).lower()
    covered = sum(1 for kw in keyword_list if kw.lower() in haystack)
    return covered / len(keyword_list)


def evaluate(cases: Iterable[dict]) -> dict:
    """汇总一批用例。每个 case 需要 retrieved / expected / keywords_retrieved 三项。"""
    case_list = list(cases)
    if not case_list:
        return {"n": 0, "hit_rate": 0.0, "recall": 0.0, "mrr": 0.0, "keyword_coverage": 0.0}

    hits = [hit_rate(c["retrieved"], c["expected"]) for c in case_list]
    recalls = [recall(c["retrieved"], c["expected"]) for c in case_list]
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
        "mrr": round(sum(reciprocals) / n, 3),
        "keyword_coverage": round(sum(coverage) / n, 3),
    }
