"""answer_eval.py 里纯函数的离线用例：整理逐题证据、取样、汇总、渲染。

建库和生成都要打百炼，这里一步都不去碰——这几步出错的话，真跑出来的数字会直接骗人。
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "knowledge-rag" / "src"))

from usage import TokenUsageCollector
from cmrc_retrieval_eval import summarize
from answer_eval import (
    _clip,
    _negative_hard_answer_line,
    build_cost,
    condition_report,
    fmt_cost,
    head_mixed,
    probe_items,
    row_from_result,
    run_condition,
    to_markdown,
)


def llm_result(input_tokens: int, output_tokens: int):
    """够用的替身：真实字段提取在 test_usage.py 里用真 AIMessage 测过了，这里只管接线。"""
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    )
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output=None)


class StubChain:
    """替身链：作答路径通过 config 里的回调发一次 on_llm_end，拒答路径什么都不发。

    这正是线上那条链的行为差别所在——拒答时不构建管道、不调模型，所以回调也不会响。
    """

    def __init__(self, results: dict):
        self.results = results
        self.seen_configs = []

    def invoke(self, question, config=None):
        self.seen_configs.append(config)
        result = self.results[question]
        if not result["refused"]:
            for handler in (config or {}).get("callbacks", []):
                handler.on_llm_end(llm_result(1500, 40))
        return result


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

class TestCostWiring:
    """成本计量接线：token 从回调进来、延迟逐题记、平均按生成次数分母。"""

    @staticmethod
    def answered_row(qid="A1", calls=1, input_tokens=1500, output_tokens=40):
        return row_from_result(
            item(qid=qid),
            {"answer": "北京", "source_documents": [doc("北京是首都", 0.2)], "refused": False},
            latency=1.234,
            usage={
                "llm_calls": calls,
                "input_tokens": input_tokens * calls,
                "output_tokens": output_tokens * calls,
                "total_tokens": (input_tokens + output_tokens) * calls,
            },
        )

    @staticmethod
    def refused_row(qid="N1"):
        item_ = item(answerable=False, golds=(), qid=qid)
        return row_from_result(
            item_,
            {"answer": "知识库里没有与这个问题足够相关的内容，不作答。", "source_documents": [doc("无关", 1.7)], "refused": True},
            latency=0.11,
        )

    def test_row_defaults_cost_fields_to_zero(self):
        """没传 usage/latency 时按 0 记：这两项「没测到」和「确实是 0」在数字上无法区分，
        所以调用方（run_condition）必须传，缺省值只服务于不关心成本的旧用例。"""
        row = row_from_result(item(), {"answer": "北京", "source_documents": [doc("北京", 0.2)], "refused": False})
        assert row["llm_calls"] == 0 and row["input_tokens"] == 0 and row["latency_seconds"] is None

    def test_run_condition_counts_generations_from_callbacks(self):
        """生成次数来自实测回调，不再由「没被阈值拦下」推断——推断的数只证明我们以为调了。"""
        answered = item(qid="A1")
        refused = item(answerable=False, golds=(), qid="N1")
        refused["question"] = "语料里根本没有的那个东西？"
        chain = StubChain({
            answered["question"]: {
                "answer": "北京",
                "source_documents": [doc("北京是首都", 0.2)],
                "refused": False,
            },
            refused["question"]: {
                "answer": "知识库里没有与这个问题足够相关的内容，不作答。",
                "source_documents": [doc("无关", 1.7)],
                "refused": True,
            },
        })

        rows, cost = run_condition(chain, [answered, refused])

        assert cost["questions"] == 2 and cost["llm_generations"] == 1
        assert cost["input_tokens"] == 1500 and cost["output_tokens"] == 40
        assert cost["mean_input_tokens_per_generation"] == 1500.0
        assert rows[0]["llm_calls"] == 1 and rows[1]["llm_calls"] == 0
        assert rows[1]["latency_seconds"] is not None, "拒答的题也要记延迟，那条路径只花检索的钱"
        assert cost["latency_generated"]["n"] == 1 and cost["latency_all"]["n"] == 2
        assert isinstance(chain.seen_configs[0]["callbacks"][0], TokenUsageCollector)

    def test_multi_call_question_averages_per_call(self):
        """一道题发几次生成时，平均值按「次」而不是按「题」算——以后 Mini Agent 多轮循环要用同一条口径。"""
        cost = build_cost([self.answered_row(calls=3)], 9.9)
        assert cost["llm_generations"] == 3
        assert cost["mean_input_tokens_per_generation"] == 1500.0

    def test_no_generation_gives_none_average_not_zero(self):
        rows = [self.refused_row()]
        cost = build_cost(rows, 0.3)

        assert cost["llm_generations"] == 0
        assert cost["mean_input_tokens_per_generation"] is None
        assert cost["mean_output_tokens_per_generation"] is None
        assert cost["latency_generated"]["n"] == 0, "没调模型就不该有「生成题延迟」"
        assert cost["latency_all"]["n"] == 1

    def test_fmt_cost_renders_tokens_latency_and_total(self):
        line = fmt_cost(build_cost([self.answered_row()], 3.2))
        assert "生成 1 次" in line and "合计 1540 tok" in line and "每次生成平均输入 1500.0 tok" in line
        assert "生成题延迟 中位 1.234s" in line and "总耗时 3.2s" in line

    def test_fmt_cost_drops_absent_latency_instead_of_showing_zero(self):
        line = fmt_cost(build_cost([self.refused_row()], 0.3))
        assert "没有发生生成" in line
        assert "生成题延迟" not in line, "n=0 时渲染出「中位 0.0s」会被读成测得很准"


class TestNoShadowedDefinitions:
    """同名函数写两遍时，后一份会静默盖掉前一份，用例照样全绿、跑出来却是旧行为。

    加成本列那轮就真的把 fmt_cost 粘贴重复了一次，靠这条用例兜住。
    """

    def test_answer_eval_has_no_duplicate_top_level_names(self):
        tree = ast.parse((Path(__file__).resolve().parent / "answer_eval.py").read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        dupes = sorted({name for name in names if names.count(name) > 1})
        assert dupes == [], f"顶层重复定义会被后一份覆盖：{dupes}"

class TestSummarizeQuantiles:
    """分位数工具是三个评测脚本共用的，n 小的时候绝不能渲染出反序数字。

    旧实现下标向下取整，n=2 时 p75 == min，成本行就印成「中位 7.4s / p75 3.3s」，
    看着像统计炸了。
    """

    def test_two_samples_stay_monotonic(self):
        stats = summarize([11.526, 3.292])

        assert stats["p25"] <= stats["median"] <= stats["p75"], stats
        assert stats["p75"] == 11.526

    def test_four_samples_pick_nearest_instead_of_collapsing_to_min(self):
        """n=4 时旧实现的 p25 就是 min（下标 0），那不成其为分位数了。"""
        stats = summarize([1.0, 2.0, 3.0, 4.0])

        assert (stats["min"], stats["p25"], stats["median"], stats["p75"], stats["max"]) == (
            1.0, 2.0, 2.5, 3.0, 4.0
        )

    def test_empty_reports_zero_instead_of_crashing(self):
        assert summarize([]) == {"n": 0}
