"""答案层指标的离线用例：全部是纯函数比对，不打模型、不联网。"""

from answer_metrics import (
    REFUSAL_MAX_CHARS,
    abstention_stats,
    answer_scores,
    char_f1,
    exact_match,
    faithfulness,
    looks_like_refusal,
    normalize_cn,
    refused,
    span_containment,
)


class TestNormalize:
    def test_drops_punctuation_and_spaces(self):
        assert normalize_cn("北京，是首都。 ") == "北京是首都"

    def test_fullwidth_becomes_halfwidth_and_lowercased(self):
        assert normalize_cn("１２３ＡＢ") == "123ab"

    def test_keeps_symbols_that_carry_meaning(self):
        # S 类（~ % 等）不丢：答案 span 里可能真的带
        assert normalize_cn("7~9片") == "7~9片"

    def test_empty(self):
        assert normalize_cn("") == ""
        assert normalize_cn(None) == ""


class TestExactMatch:
    def test_punctuation_only_difference_still_matches(self):
        assert exact_match("北京，是首都。", ["北京是首都"]) == 1.0

    def test_any_gold_answer_counts(self):
        assert exact_match("山脊", ["山谷", "山脊"]) == 1.0

    def test_rewrites_do_not_match(self):
        """gold 是短 span，模型改写成整句时 EM 判 0 —— 所以 EM 不能单独看。"""
        assert exact_match("首都是北京。", ["北京"]) == 0.0

    def test_empty_prediction(self):
        assert exact_match("", ["北京"]) == 0.0


class TestCharF1:
    def test_identical(self):
        assert char_f1("abc", "abc") == 1.0

    def test_repeated_chars_use_multiset_not_set(self):
        # 多重集合：common=1，p=1/3，r=1 → F1=0.5；按集合算会错成 1.0
        assert char_f1("哈哈哈", "哈") == 0.5

    def test_partial_overlap(self):
        assert round(char_f1("abc", "abd"), 4) == 0.6667

    def test_no_overlap_and_empty(self):
        assert char_f1("abc", "xyz") == 0.0
        assert char_f1("", "abc") == 0.0
        assert char_f1("abc", "") == 0.0

    def test_answer_scores_picks_best_gold(self):
        result = answer_scores("北京是首都", ["上海", "北京"])
        assert result["em"] == 0.0
        assert result["f1"] == round(char_f1("北京是首都", "北京"), 4)


class TestSpanContainment:
    def test_span_inside_a_full_sentence_counts(self):
        """冒烟实测的形状：gold 是"赛马"，模型答"…是香港赛马的锦标赛"，EM 0 但内容是对的。"""
        assert span_containment("香港冠军暨遮打杯是香港赛马的锦标赛。", ["赛马"]) == 1.0

    def test_no_span(self):
        assert span_containment("这是一个错误的答案", ["赛马"]) == 0.0

    def test_any_gold_counts_and_empty_inputs(self):
        assert span_containment("通行于黑龙江流域", ["松花江", "黑龙江"]) == 1.0
        assert span_containment("", ["北京"]) == 0.0
        assert span_containment("北京", []) == 0.0

    def test_answer_scores_reports_all_three(self):
        scores = answer_scores("首都是北京。", ["北京"])
        assert scores["em"] == 0.0
        assert scores["containment"] == 1.0
        assert 0 < scores["f1"] < 1.0


class TestFaithfulness:
    CONTEXT = ["上海人口为2400万人，居全国首位"]

    def test_unigram_saturates_but_bigram_catches_the_gap(self):
        """每个字都在上下文里（unigram 1.0），但"口2"这个二字组不存在——编造就是这么露馅的。"""
        result = faithfulness("上海人口2400万", self.CONTEXT)
        assert result["unigram"] == 1.0
        assert result["bigram"] < 1.0

    def test_verbatim_span_scores_full(self):
        assert faithfulness("上海人口为2400万人", self.CONTEXT)["bigram"] == 1.0

    def test_fabricated_text_scores_low(self):
        assert faithfulness("北京人口众多繁华", self.CONTEXT)["bigram"] < 0.3

    def test_fabricated_numbers_are_blunter_because_digits_coincide(self):
        """二字组对编造的数字比较钝：'00' 真的出现在 2400 里，所以只掉到 0.5 而不是 0。"""
        assert faithfulness("北京人口9000万", self.CONTEXT)["bigram"] == 0.5

    def test_empty_context_and_single_char_answer(self):
        assert faithfulness("北京", [])["unigram"] == 0.0
        single = faithfulness("京", self.CONTEXT)
        assert single["bigram"] == 0.0 and single["n_chars"] == 1

    def test_empty_answer(self):
        assert faithfulness("", self.CONTEXT) == {"unigram": 0.0, "bigram": 0.0, "n_chars": 0}


class TestLooksLikeRefusal:
    def test_short_marker_hit(self):
        assert looks_like_refusal("知识库里没有相关信息，无法回答。") is True
        assert looks_like_refusal("I don't know.") is True

    def test_strong_marker_ignores_length(self):
        """这三条是冒烟跑出来的真实输出，早期 30 字上限把它们全漏了。"""
        real = [
            "上下文未提供关于“鑫诺一号通信卫星”制造公司的任何信息。",
            "上下文未提供关于佩维斯·埃里森（Pervis Ellison）的任何信息，因此无法回答他在场上的位置。",
            "根据提供的上下文，没有提到“胸斑眶锯雀鲷”这一名称，也没有其任何别名或俗名信息。"
            "因此，无法从给定材料中确定它又可以叫作什么。",
        ]
        for text in real:
            assert looks_like_refusal(text) is True, text

    def test_weak_marker_needs_a_short_reply(self):
        """弱标记（无法确定 / 未给出）常出现在正常答案里，超过 30 字就不当拒答。"""
        long_answer = (
            "根据上下文，这个项目的总预算无法确定，因为它分成三个阶段估算，"
            "其中第二阶段包含设备采购与人员培训费用，合计约一千二百万元。"
        )
        assert len(normalize_cn(long_answer)) > REFUSAL_MAX_CHARS
        assert looks_like_refusal(long_answer) is False

        hedged = (
            "根据上下文，该指标的具体数值未给出，但整体趋势与上表一致，"
            "第二季度环比增长约十二个百分点，全年增幅则略低于上年。"
        )
        assert len(normalize_cn(hedged)) > REFUSAL_MAX_CHARS
        assert looks_like_refusal(hedged) is False
        assert looks_like_refusal("这个数值上下文里未给出。") is True

    def test_plain_answer_is_not_refusal(self):
        assert looks_like_refusal("上海人口为2400万人") is False


class TestRefused:
    def test_either_layer_counts(self):
        assert refused({"store_refused": True, "model_self_refused": False}) is True
        assert refused({"store_refused": False, "model_self_refused": True}) is True
        assert refused({"store_refused": False, "model_self_refused": False}) is False


def make_positive(em=1.0, f1=1.0, containment=1.0, store_refused=False, self_refused=False):
    return {
        "answerable": True,
        "store_refused": store_refused,
        "model_self_refused": self_refused,
        "em": em,
        "f1": f1,
        "containment": containment,
        "faithfulness_unigram": 1.0,
        "faithfulness_bigram": 0.8,
    }


def make_negative(store_refused=False, self_refused=False, negative_f1=0.0):
    return {
        "answerable": False,
        "store_refused": store_refused,
        "model_self_refused": self_refused,
        "negative_f1": negative_f1,
    }


class TestAbstentionStats:
    def test_two_denominators_expose_refusal_squeezing_the_sample(self):
        """2 题答对、2 题被拒：只算作答子集是 1.0，端到端口径是 0.5——差的就是被拿走的分母。"""
        rows = [make_positive(), make_positive(),
                make_positive(store_refused=True), make_positive(self_refused=True)]
        stats = abstention_stats(rows)

        assert stats["quality_answered_subset"]["n"] == 2
        assert stats["quality_answered_subset"]["f1"] == 1.0
        assert stats["quality_answered_subset"]["containment"] == 1.0
        assert stats["quality_all_answerable_zero_refused"]["n"] == 4
        assert stats["quality_all_answerable_zero_refused"]["f1"] == 0.5
        assert stats["quality_all_answerable_zero_refused"]["containment"] == 0.5
        assert stats["answerable"]["either_refused"] == 2

    def test_containment_separates_right_content_from_right_format(self):
        """整句里含着正确 span：EM 0、span 命中 1——质量分不能只按 EM 报。"""
        rows = [make_positive(em=0.0, f1=0.3, containment=1.0)]
        stats = abstention_stats(rows)

        assert stats["quality_answered_subset"]["em"] == 0.0
        assert stats["quality_answered_subset"]["containment"] == 1.0

    def test_refusal_layers_counted_separately_and_together(self):
        rows = [
            make_negative(store_refused=True),
            make_negative(store_refused=True, self_refused=True),
            make_negative(self_refused=True),
            make_negative(negative_f1=0.9),
        ]
        stats = abstention_stats(rows)
        negative = stats["negative"]

        assert negative["store_refused"] == 2
        assert negative["model_self_refused"] == 2
        assert negative["either_refused"] == 3
        assert negative["both_refused"] == 1
        assert negative["answered"] == 1
        assert negative["refusal_rate"] == 0.75
        # 唯一被硬答的那道 F1 0.9，说明答案来自模型参数而不是检索，不能算链路正确
        assert stats["negative_hard_answered_matches_gold"] == 1
        assert stats["negative_hard_answered_f1"] == 0.9

    def test_all_refused_does_not_hide_behind_a_clean_subset(self):
        stats = abstention_stats([make_positive(store_refused=True) for _ in range(3)])
        assert stats["quality_answered_subset"]["n"] == 0
        assert stats["quality_answered_subset"]["f1"] == 0.0
        assert stats["quality_all_answerable_zero_refused"]["f1"] == 0.0

    def test_empty_rows_do_not_crash(self):
        stats = abstention_stats([])
        assert stats["answerable"]["n"] == 0 and stats["negative"]["refusal_rate"] == 0.0
