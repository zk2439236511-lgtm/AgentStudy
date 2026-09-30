"""答案层指标：纯函数，不联网、不打模型，可以被 pytest 全覆盖。

三层各判什么，别混：
- exact_match / char_f1：答得对不对。CMRC 的 gold 是原文里的短 span（平均约 10 字），
  模型若改写成完整句子，EM 会直接判 0，所以 EM 与 F1 必须一起看。
- faithfulness：答案有没有被**检索到的上下文**撑住。这里只做字面支撑率，
  不是事实性判定——上下文里说错的东西它照样给高分。
- abstention：不该答的时候有没有停。这是分类问题，走混淆矩阵，
  被拒答的题**不进 EM 的分母**（否则"全都拒答"能在干净分母上刷出高平均分）。

F1 与去标点口径参照 CMRC2018 官方评测脚本的思路（去标点空白、按字符算），
但这是简化复刻，不是逐字节等价，别把它当成榜单可比的分。
"""

import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

# 拒答判定分两档，界线是用冒烟跑出来的真实输出校准的：
# - 强标记：句子本身就是"上下文里没有这条信息"的说法，出现即算自拒，不看长度。
#   早期只设了一条 30 字上限，实测模型写的是"上下文未提供关于“鑫诺一号通信卫星”制造公司的
#   任何信息。"——整段超过 30 字，3 道真实自拒全被漏判，再往下跑就会得出"提示词层兜不住"
#   这个错结论。判据错了比数字难看更可怕，所以先修判据。
# - 弱标记："无法确定""不包含"这类也常出现在正常答案里（转述上下文的不确定表述），
#   只有短回复才当拒答。取向仍是宁可漏判弱标记，也别把正常答案误判成拒答。
STRONG_REFUSAL_MARKERS = (
    "无法回答", "无法作答", "不能回答", "未提供", "没有提供", "没有提到", "没有提及",
    "未提及", "没有相关信息", "没有涉及", "无法从上下文", "无法从给定", "知识库里没有",
    "不知道", "cannot answer", "no information", "not mentioned", "unable to answer",
    "don't know",
)
WEAK_REFUSAL_MARKERS = ("不包含", "无法确定", "没有给出", "未给出")
REFUSAL_MAX_CHARS = 30


def normalize_cn(text: str) -> str:
    """NFKC 归一（全角数字字母转半角）+ 去标点与空白 + 小写。

    只丢 P（标点）和 Z（分隔符）两类：S 里的符号留着，因为答案 span 可能带 ~、% 这种。
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    kept = [ch for ch in folded if not unicodedata.category(ch)[0] in ("P", "Z")]
    return "".join(kept).lower()


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    """归一化后与**任一** gold 完全相等即算命中（CMRC 每题有 1~2 个人工答案）。"""
    normalized = normalize_cn(prediction)
    if not normalized:
        return 0.0
    return 1.0 if any(normalized == normalize_cn(gold) for gold in gold_answers) else 0.0


def char_f1(prediction: str, gold: str) -> float:
    """字符多重集合的 F1：重复的字要按次数计，"哈哈哈"对上"哈"不该是满分。"""
    pred, target = normalize_cn(prediction), normalize_cn(gold)
    if not pred or not target:
        return 0.0
    common = sum((Counter(pred) & Counter(target)).values())
    if not common:
        return 0.0
    precision = common / len(pred)
    recall = common / len(target)
    return 2 * precision * recall / (precision + recall)


def span_containment(prediction: str, gold_answers: Sequence[str]) -> float:
    """正确 span 有没有出现在答案里。

    EM 要求整句等于 span，模型把答案写成完整句子时 EM 恒为 0（冒烟实测：5 道作答全 EM 0，
    但 F1 0.21~0.89，答案里其实含着正确的词）。这条判据补的就是这个缺口。
    """
    normalized = normalize_cn(prediction)
    if not normalized:
        return 0.0
    return 1.0 if any(
        normalize_cn(gold) and normalize_cn(gold) in normalized for gold in gold_answers
    ) else 0.0


def answer_scores(prediction: str, gold_answers: Sequence[str]) -> dict:
    """对一题给 EM、F1 和 span 命中；F1 取所有 gold 里最好的那个（人工标注等价答案）。"""
    return {
        "em": exact_match(prediction, gold_answers),
        "f1": round(max((char_f1(prediction, gold) for gold in gold_answers), default=0.0), 4),
        "containment": span_containment(prediction, gold_answers),
    }


def faithfulness(prediction: str, context_texts: Iterable[str]) -> dict:
    """答案字面有多少比例来自检索到的上下文。

    unigram 会虚高：常用字在长上下文里几乎必然出现。bigram（相邻二字组）才测得出
    "答案的片段是不是成串来自上下文"，编造出来的东西在 bigram 上先露馅。两个都记。
    """
    answer = normalize_cn(prediction)
    haystack = normalize_cn("".join(context_texts))
    if not answer:
        return {"unigram": 0.0, "bigram": 0.0, "n_chars": 0}

    unigram = sum(1 for ch in answer if ch in haystack) / len(answer)
    bigrams = [answer[i : i + 2] for i in range(len(answer) - 1)]
    bigram = sum(1 for pair in bigrams if pair in haystack) / len(bigrams) if bigrams else 0.0
    return {"unigram": round(unigram, 4), "bigram": round(bigram, 4), "n_chars": len(answer)}


def looks_like_refusal(text: str) -> bool:
    """模型自己的拒答（提示词里那句 "say so" 走的就是这条路）。

    英文标记带撇号，归一化后 "don't know" 会变成 "dontknow"，所以原文和归一化文本都要比。
    """
    normalized = normalize_cn(text)
    if not normalized:
        return False
    raw = (text or "").lower()

    def hit(marker: str) -> bool:
        return marker in raw or marker in normalized

    if any(hit(marker) for marker in STRONG_REFUSAL_MARKERS):
        return True
    if len(normalized) > REFUSAL_MAX_CHARS:
        return False
    return any(hit(marker) for marker in WEAK_REFUSAL_MARKERS)


def refused(row: dict) -> bool:
    """任一拦截层拒了：检索层阈值（权威，写在 refused 字段里）或模型自拒。"""
    return bool(row.get("store_refused") or row.get("model_self_refused"))


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def abstention_stats(rows: Sequence[dict]) -> dict:
    """把一批逐题结果算成拒答混淆矩阵 + 两个口径的答案质量。

    每题需要：answerable、store_refused、model_self_refused，可回答题还要 em、f1、
    containment，负样本还要 negative_f1（拿它自带的真实答案比出来的，用来看参数知识）。

    质量两个口径都要报：
    - answered_subset：只算真正作答的题。拒答越多分母越小，平均分越容易好看。
    - all_answerable_zero_refused：可回答题全进分母，被拒的按 0 计。这才是端到端质量。
    两者的差就是"拒答把多少题从分母里拿走了"，单看前者会自欺。
    """
    row_list = list(rows)
    positives = [row for row in row_list if row["answerable"]]
    negatives = [row for row in row_list if not row["answerable"]]

    def group(rows_of_one_kind: list[dict]) -> dict:
        store = sum(1 for row in rows_of_one_kind if row.get("store_refused"))
        model = sum(1 for row in rows_of_one_kind if row.get("model_self_refused"))
        either = sum(1 for row in rows_of_one_kind if refused(row))
        both = sum(
            1 for row in rows_of_one_kind
            if row.get("store_refused") and row.get("model_self_refused")
        )
        return {
            "n": len(rows_of_one_kind),
            "store_refused": store,
            "model_self_refused": model,
            "either_refused": either,
            "both_refused": both,
            "answered": len(rows_of_one_kind) - either,
            "refusal_rate": round(either / len(rows_of_one_kind), 4) if rows_of_one_kind else 0.0,
        }

    answered_positives = [row for row in positives if not refused(row)]
    hard_answered_negatives = [row for row in negatives if not refused(row)]

    return {
        "answerable": group(positives),
        "negative": group(negatives),
        "quality_answered_subset": {
            "n": len(answered_positives),
            "em": _mean([float(row.get("em", 0.0)) for row in answered_positives]),
            "f1": _mean([float(row.get("f1", 0.0)) for row in answered_positives]),
            "containment": _mean([
                float(row.get("containment", 0.0)) for row in answered_positives
            ]),
            "faithfulness_unigram": _mean([
                float(row.get("faithfulness_unigram", 0.0)) for row in answered_positives
            ]),
            "faithfulness_bigram": _mean([
                float(row.get("faithfulness_bigram", 0.0)) for row in answered_positives
            ]),
        },
        "quality_all_answerable_zero_refused": {
            "n": len(positives),
            "em": _mean([
                0.0 if refused(row) else float(row.get("em", 0.0)) for row in positives
            ]),
            "f1": _mean([
                0.0 if refused(row) else float(row.get("f1", 0.0)) for row in positives
            ]),
            "containment": _mean([
                0.0 if refused(row) else float(row.get("containment", 0.0)) for row in positives
            ]),
        },
        # 负样本被硬答时，与它自带的真实答案比对：F1 高说明答案来自模型参数而不是检索，
        # 这不代表链路正确，只是"蒙对了"，要单独看不能混进质量分。
        "negative_hard_answered_f1": _mean(
            [float(row.get("negative_f1", 0.0)) for row in hard_answered_negatives]
        ),
        "negative_hard_answered_matches_gold": sum(
            1 for row in hard_answered_negatives if float(row.get("negative_f1", 0.0)) >= 0.5
        ),
    }
