"""answer_eval.py 里纯函数的离线用例：整理逐题证据、取样、汇总、渲染。

建库和生成都要打百炼，这里一步都不去碰——这几步出错的话，真跑出来的数字会直接骗人。
"""

import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "knowledge-rag" / "src"))

from answer_eval import (
    _clip,
    _negative_hard_answer_line,
    condition_report,
    head_mixed,
    probe_items,
    row_from_result,
    to_markdown,
)


def doc(text: str, score=None) -> Document:
    metadata = {"doc_id": "D1"}
    if score is not None:
        metadata["score"] = score
    return Document(page_content=text, metadata=metadata)


def item(answerable=True, golds=("北京",), qid="A1"):
    return {
        "question_id": qid,
        "question": "首都？",
        "answerable": answerable,
        "gold_doc_ids": ["D1"] if answerable else [],
        "gold_answers": list(golds),
    }


class TestRowFromResult:
    def test_answered_positive_gets_scores(self):
        row = row_from_result(
            item(),
            {"answer": "北京", "source_documents": [doc("北京是首都", 0.25)], "refused": False},
        )
        assert row["em"] == 1.0 and row["f1"] == 1.0
        assert row["containment"] == 1.0
        assert row["top1_distance"] == 0.25
        assert row["faithfulness_unigram"] == 1.0 and row["faithfulness_bigram"] == 1.0
        assert row["store_refused"] is False and row["model_self_refused"] is False

    def test_store_refusal_leaves_quality_fields_as_none(self):
        """阈值拒答时 answer 是我们自己写的文案，拿它去和金标 span 比分会算出莫名其妙的数字。"""
        refusal = "知识库里没有与这个问题足够相关的内容，不作答。（最接近的片段距离 1.70，拒答阈值 1.00）"
        row = row_from_result(item(), {"answer": refusal, "source_documents": [doc("无关段落", 1.7)], "refused": True})

        assert row["store_refused"] is True
        assert row["model_self_refused"] is False, "自家文案不能被判成模型自拒"
        assert row["em"] is None and row["f1"] is None and row["containment"] is None
        assert row["faithfulness_unigram"] is None and row["faithfulness_bigram"] is None

    def test_model_self_refusal_is_detected_only_when_store_let_it_through(self):
        row = row_from_result(
            item(),
            {"answer": "上下文里没有提到这个。", "source_documents": [doc("无关段落", 0.4)], "refused": False},
        )
        assert row["model_self_refused"] is True
        assert row["em"] == 0.0 and row["f1"] == 0.0

    def test_negative_hard_answer_is_scored_against_withheld_gold(self):
        """负样本自带的 gold 来自被扣住的那段：答得上是模型参数里的知识，不是检索立功。"""
        row = row_from_result(
            item(answerable=False, golds=("147位",), qid="N1"),
            {"answer": "147位", "source_documents": [doc("无关段落", 1.2)], "refused": False},
        )
        assert row["em"] is None and row["f1"] is None
        assert row["negative_f1"] == 1.0

    def test_missing_score_gives_none_distance(self):
        row = row_from_result(
            item(),
            {"answer": "北京", "source_documents": [doc("北京是首都")], "refused": False},
        )
        assert row["top1_distance"] is None

    def test_distance_is_rounded_not_truncated(self):
        row = row_from_result(
            item(),
            {"answer": "北京", "source_documents": [doc("北京", 0.123456)], "refused": False},
        )
        assert row["top1_distance"] == 0.1235


class TestProbeItems:
    GOLDEN = (
        [{"question_id": f"N{i}", "answerable": False} for i in range(2)]
        + [{"question_id": f"P{i}", "answerable": True} for i in range(10)]
    )

    def test_zero_budget_keeps_only_negatives(self):
        assert [item["question_id"] for item in probe_items(self.GOLDEN, 0)] == ["N0", "N1"]

    def test_stride_sampling_is_spread_and_deterministic(self):
        sample = probe_items(self.GOLDEN, 4)
        assert [item["question_id"] for item in sample] == ["N0", "N1", "P0", "P2", "P5", "P7"]
        assert probe_items(self.GOLDEN, 4) == sample

    def test_budget_larger_than_the_group_takes_everything(self):
        assert len(probe_items(self.GOLDEN, 99)) == 12


class TestHeadMixed:
    ITEMS = (
        [{"question_id": f"N{i}", "answerable": False} for i in range(5)]
        + [{"question_id": f"P{i}", "answerable": True} for i in range(5)]
    )

    def test_keeps_both_paths_in_a_small_sample(self):
        """冒烟取样要同时出现拒答与作答，只切前 N 条会全是负样本、生成路径一次都不走。"""
        assert [item["question_id"] for item in head_mixed(self.ITEMS, 6)] == [
            "N0", "N1", "N2", "P0", "P1", "P2"
        ]

    def test_zero_and_negative_budget_is_empty(self):
        assert head_mixed(self.ITEMS, 0) == [] and head_mixed(self.ITEMS, -1) == []

    def test_more_than_available_capped_by_positives(self):
        assert len(head_mixed(self.ITEMS, 99)) == 8  # 3 道负样本 + 全部 5 道可回答题


class TestConditionReport:
    def test_cost_passthrough_and_f1_distribution_counts_only_answered(self):
        rows = [
            row_from_result(item(qid="A1"), {"answer": "北京", "source_documents": [doc("北京", 0.2)], "refused": False}),
            row_from_result(item(qid="A2"), {"answer": "错误答案", "source_documents": [doc("北京", 0.2)], "refused": False}),
            row_from_result(item(qid="A3"), {"answer": "不作答", "source_documents": [doc("北京", 1.8)], "refused": True}),
        ]
        report = condition_report(rows, {"questions": 3, "llm_generations": 2})

        assert report["cost"] == {"questions": 3, "llm_generations": 2}
        assert report["f1_distribution"]["n"] == 2
        assert report["quality_answered_subset"]["n"] == 2
        assert report["quality_all_answerable_zero_refused"]["n"] == 3


class TestToMarkdown:
    def build_report(self):
        on_rows = [
            row_from_result(item(qid="A1"), {"answer": "北京", "source_documents": [doc("北京是首都", 0.2)], "refused": False}),
            row_from_result(item(answerable=False, golds=("147位",), qid="N1"),
                            {"answer": "不作答", "source_documents": [doc("无关", 1.7)], "refused": True}),
        ]
        off_rows = [
            row_from_result(item(answerable=False, golds=("147位",), qid="N1"),
                            {"answer": "147位", "source_documents": [doc("无关", 1.7)], "refused": False}),
        ]
        return {
            "chat_model": "qwen-plus",
            "k": 8,
            "chunk_size": 0,
            "max_distance": 1.0,
            "conditions": {
                "threshold_on": condition_report(on_rows, {"questions": 2, "llm_generations": 1}),
                "no_threshold_probe": condition_report(off_rows, {"questions": 1, "llm_generations": 1}),
            },
            "rows": {"threshold_on": on_rows, "no_threshold_probe": off_rows},
        }

    def test_renders_every_section(self):
        markdown = to_markdown(self.build_report())
        for heading in ("## 1.", "## 2.", "## 3.", "### 4.1", "### 4.2", "## 5."):
            assert heading in markdown
        assert "d ≤ 1.0" in markdown
        # 负样本原文要能肉眼看到，拒答判定不能只给一个数字
        assert "N1" in markdown and "147位" in markdown

    def test_f1_distribution_line_names_its_own_denominator(self):
        """分布行的 n 比作答子集大（模型自拒的可回答题记 0 分参与判分），标签要把这个说清楚。"""
        report = self.build_report()
        extra = row_from_result(
            item(qid="A2"),
            {"answer": "上下文未提及这一点。", "source_documents": [doc("无关", 0.4)], "refused": False},
        )
        report["rows"]["threshold_on"].append(extra)
        report["conditions"]["threshold_on"] = condition_report(
            report["rows"]["threshold_on"], {"questions": 3, "llm_generations": 3}
        )

        markdown = to_markdown(report)
        assert "n=2" in markdown and "作答 1" in markdown and "模型自拒记 0 分的 1" in markdown
        assert "作答子集 char-F1 分布" not in markdown

    def test_does_not_crash_without_negatives_or_answers(self):
        report = self.build_report()
        report["rows"]["threshold_on"] = []
        report["conditions"]["threshold_on"] = condition_report([], {"questions": 0})
        report["conditions"]["no_threshold_probe"] = condition_report([], {"questions": 0})
        report["rows"]["no_threshold_probe"] = []

        assert "## 5." in to_markdown(report)


class TestNegativeHardAnswerLine:
    def test_no_hard_answer_says_no_data_instead_of_f1_zero(self):
        """一道都没硬答时不能照打 F1=0.0：那会被读成「答得很差」，实际是没有样本可比。"""
        rows = [row_from_result(item(answerable=False, qid="N1"),
                                 {"answer": "不作答", "source_documents": [doc("无关", 1.7)], "refused": True})]
        line = _negative_hard_answer_line(condition_report(rows, {"questions": 1}))
        assert "没有硬答" in line and "0.0" not in line

    def test_hard_answer_reports_the_parametric_knowledge_signal(self):
        rows = [row_from_result(item(answerable=False, golds=("147位",), qid="N1"),
                                {"answer": "147位", "source_documents": [doc("无关", 1.7)], "refused": False})]
        line = _negative_hard_answer_line(condition_report(rows, {"questions": 1}))
        assert "被硬答 1 道" in line and "1.0" in line


class TestClip:
    def test_escapes_pipes_and_truncates(self):
        assert _clip("a|b") == "a\\|b"
        assert _clip("一二三四五", 3) == "一二三…"
        assert _clip("") == ""
