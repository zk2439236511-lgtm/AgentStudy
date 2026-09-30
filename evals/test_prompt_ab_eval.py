"""`prompt_ab_eval.py` 的离线用例：选题、配对、判据、渲染。

这个脚本唯一的作用是"只改一个变量然后判它有没有用"，所以用例全部围着三件事：
分层选题不能把拒答的题选进来、配对差值算得对、判据在被改动"表面成功"时必须能说不。
真调百炼的那部分不在这里测。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_ab_eval import (
    CONCISE_TEMPLATE,
    answered_rows,
    budget,
    decide,
    items_for,
    paired,
    quality,
    select_strata,
    to_markdown,
)
from ragdemo.rag_chain import RAG_TEMPLATE


def make_row(qid: str, f1, predicted: str, *, containment=1.0, answerable=True,
             em=0.0, bigram=0.9, output_tokens=20, gold=("标准答案",), stratum="mid"):
    return {
        "question_id": qid,
        "answerable": answerable,
        "f1": f1,
        "em": em,
        "containment": containment,
        "faithfulness_bigram": bigram,
        "predicted": predicted,
        "output_tokens": output_tokens,
        "gold_answers": list(gold),
        "stratum": stratum,
    }


class TestTemplateIsSingleVariable:
    """改动只允许多出"简短作答"那一句，其余逐字保留——否则实验变量不止一个。"""

    def test_keeps_the_two_leading_instruction_lines_verbatim(self):
        head = "\n".join(RAG_TEMPLATE.splitlines()[:2])
        assert CONCISE_TEMPLATE.startswith(head), "前两句（只依据上下文 + 允许说不知道）必须原样"

    def test_keeps_the_prompt_skeleton(self):
        assert CONCISE_TEMPLATE.endswith("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")

    def test_only_adds_the_brevity_block(self):
        assert len(CONCISE_TEMPLATE.splitlines()) > len(RAG_TEMPLATE.splitlines())
        assert "briefly" in CONCISE_TEMPLATE


class TestAnsweredRows:
    def test_drops_refused_and_negatives(self, tmp_path):
        report = {
            "rows": {
                "threshold_on": [
                    make_row("A", 0.5, "答案"),
                    make_row("B", None, "知识库里没有…（拒答文案）"),
                    make_row("C", 0.4, "负样本硬答", answerable=False),
                ]
            }
        }
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

        assert [row["question_id"] for row in answered_rows(path)] == ["A"]


class TestSelectStrata:
    def build_pool(self):
        rows = [make_row(f"L{i}", 0.1 + i * 0.01, f"低分答案{i}" * (10 - i)) for i in range(6)]
        rows += [make_row(f"M{i}", 0.4 + i * 0.04, f"中分答案{i}" * (5 - i)) for i in range(5)]
        rows += [make_row(f"H{i}", 1.0, f"满分{i}字" * (i + 1)) for i in range(4)]
        rows.append(make_row("X", 0.35, "落在三档缝隙里的题"))
        return rows

    def test_strata_counts_and_order(self):
        picked = select_strata(self.build_pool(), 3, 2, 2)

        assert [row["stratum"] for row in picked] == ["low"] * 3 + ["mid"] * 2 + ["high"] * 2

    def test_low_takes_the_worst_first(self):
        picked = select_strata(self.build_pool(), 3, 0, 0)

        assert [row["question_id"] for row in picked] == ["L0", "L1", "L2"]

    def test_mid_prefers_the_longest_answers(self):
        """本轮假设是"啰嗦冲淡了 F1"，中段当然要先看最啰嗦的那几道。"""
        picked = select_strata(self.build_pool(), 0, 2, 0)

        assert [row["question_id"] for row in picked] == ["M0", "M1"]

    def test_high_prefers_the_shortest_full_marks(self):
        picked = select_strata(self.build_pool(), 0, 0, 2)

        assert [row["question_id"] for row in picked] == ["H0", "H1"]

    def test_no_question_appears_twice(self):
        picked = select_strata(self.build_pool(), 6, 5, 4)
        ids = [row["question_id"] for row in picked]

        assert len(ids) == len(set(ids))

    def test_asks_beyond_pool_are_dropped_not_reused(self):
        picked = select_strata(self.build_pool(), 99, 99, 99)

        assert len(picked) == 15, "池子只有 15 道可选题，不能凑数重复选"

    def test_items_for_keeps_selection_order(self):
        picked = select_strata(self.build_pool(), 2, 1, 1)
        golden = [{"question_id": f"Q{index}", "question": "问"} for index in range(50)]
        golden += [{"question_id": row["question_id"], "question": "问"} for row in picked]

        items = items_for(golden, picked)

        assert [item["question_id"] for item in items] == [row["question_id"] for row in picked]


class TestQuality:
    def test_only_scored_rows_count(self):
        rows = [make_row("A", 0.5, "abcd"), make_row("B", None, "拒答")]

        stats = quality(rows)

        assert stats["n"] == 1
        assert stats["f1"] == 0.5
        assert stats["mean_answer_chars"] == 4.0

    def test_empty_gives_none_instead_of_crashing(self):
        stats = quality([])

        assert stats["n"] == 0
        assert stats["f1"] is None and stats["mean_answer_chars"] is None


class TestPaired:
    def test_per_question_deltas(self):
        base = [make_row("A", 0.2, "很长很长的答案" * 6)]
        variant = [make_row("A", 0.6, "短答案")]

        line = paired(base, variant)[0]

        assert line["f1_before"] == 0.2 and line["f1_after"] == 0.6
        assert line["chars_before"] > line["chars_after"]
        assert line["missing"] is False

    def test_question_missing_from_variant_is_kept_as_missing_not_dropped(self):
        """这轮没生成出来（被拦下/异常）必须留痕，静默丢题会让平均值凭空变好。"""
        base = [make_row("A", 0.2, "答案"), make_row("B", 0.5, "答案")]
        variant = [make_row("A", 0.2, "答案"), make_row("B", None, "知识库里没有相关内容，不作答")]

        lines = paired(base, variant)

        assert len(lines) == 2
        assert lines[0]["missing"] is False
        assert lines[1]["missing"] is True

    def test_rows_absent_in_variant_entirely_are_skipped(self):
        base = [make_row("A", 0.2, "答案")]

        assert paired(base, []) == []


class TestDecide:
    def test_winning_case_passes_all_four(self):
        base = [make_row(f"L{i}", 0.2, "啰嗦的答案" * 10) for i in range(3)]
        base += [make_row(f"H{i}", 1.0, "对", stratum="high") for i in range(2)]
        variant = [make_row(f"L{i}", 0.7, "对") for i in range(3)]
        variant += [make_row(f"H{i}", 1.0, "对", stratum="high") for i in range(2)]

        summary, verdicts = decide(base, variant, paired(base, variant))

        assert summary["all_passed"] is True
        assert all(verdict["passed"] for verdict in verdicts)

    def test_f1_up_without_shorter_answers_fails_the_mechanism_check(self):
        """分数涨了但答案没变短 → 不是这段提示词带来的，判据必须说不。"""
        base = [make_row("Q1", 0.2, "一样长"), make_row("Q2", 0.2, "一样长")]
        variant = [make_row("Q1", 0.9, "一样长"), make_row("Q2", 0.9, "一样长")]

        _, verdicts = decide(base, variant, paired(base, variant))

        assert verdicts[0]["passed"] is True, "主判据：F1 确实涨了"
        assert verdicts[3]["passed"] is False, "机制判据：字数没降就不能归因给这段提示词"

    def test_sentinel_regression_beyond_tolerance_fails(self):
        base = [make_row("L0", 0.1, "啰" * 40)]
        base += [make_row(f"H{i}", 1.0, "对", stratum="high") for i in range(3)]
        variant = [make_row("L0", 0.8, "对")]
        variant += [make_row("H0", 1.0, "对", stratum="high"),
                    make_row("H1", 0.1, "答非所问", stratum="high"),
                    make_row("H2", None, "不知道", stratum="high")]

        _, verdicts = decide(base, variant, paired(base, variant))

        assert verdicts[2]["passed"] is False, "满分组回归 2 道 > 容差 1 道"
        assert "H1" in verdicts[2]["value"] and "H2" in verdicts[2]["value"], "要指名是哪两道回归"

    def test_containment_drop_fails_guard(self):
        base = [make_row("Q1", 0.2, "啰嗦的答案" * 10, containment=1.0)]
        variant = [make_row("Q1", 0.3, "短", containment=0.0)]

        _, verdicts = decide(base, variant, paired(base, variant))

        assert verdicts[1]["passed"] is False

    def test_variant_with_no_generation_does_not_crash(self):
        base = [make_row("Q1", 0.2, "答案")]
        variant = [make_row("Q1", None, "知识库里没有相关内容，不作答")]

        summary, verdicts = decide(base, variant, paired(base, variant))

        assert summary["all_passed"] is False
        assert "None" in verdicts[0]["value"], "无数据要显示成 None，不能悄悄当 0 算"


class TestBudgetAndRendering:
    def test_budget_counts_both_conditions(self):
        plan = budget([{"question_id": "A"}] * 12)

        assert plan["questions"] == 12 and plan["generations"] == 24
        # 单价是从真机冒烟来的，改它要连这条一起改，别让预算打印停留在旧价
        assert plan["estimated_input_tokens"] == 24 * 3000

    def build_report(self):
        base = [make_row("L0", 0.2, "啰嗦的答案" * 10, containment=1.0, em=0.0,
                         gold=("标准答案",))]
        base[0]["stratum"] = "low"
        variant = [make_row("L0", 0.7, "标准答案", containment=1.0, em=1.0,
                            gold=("标准答案",))]
        variant[0]["stratum"] = "low"
        cost = quality(base) and {
            "questions": 1, "llm_generations": 1, "retrievals": 1, "seconds": 1.0,
            "input_tokens": 2600, "output_tokens": 30, "total_tokens": 2630,
            "mean_input_tokens_per_generation": 2600.0,
            "mean_output_tokens_per_generation": 30.0,
            "latency_all": {"n": 1, "median": 0.9},
            "latency_generated": {"n": 1, "median": 0.9, "p75": 0.9, "max": 0.9},
        }
        summary, verdicts = decide(base, variant, paired(base, variant))
        return {
            "date": "2026-09-30",
            "chat_model": "qwen-plus",
            "k": 8,
            "chunk_size": 0,
            "max_distance": 1.0,
            "baseline": Path("evals/results/2026-09-30-answer.json"),
            "budget": budget([{"question_id": str(index)} for index in range(12)]),
            "strata": ["low", "mid", "high"],
            "selection": [{"question_id": "L0", "stratum": "low", "f1_baseline_record": 0.2}],
            "summary": summary,
            "verdicts": verdicts,
            "pairs": paired(base, variant),
            "cost": {"baseline": cost, "variant": cost},
        }

    def test_renders_every_section(self):
        markdown = to_markdown(self.build_report())

        for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
            assert heading in markdown
        assert "CONCISE_TEMPLATE" in markdown
        assert "24 次生成" in markdown

    def test_missing_generation_is_shown_not_hidden(self):
        report = self.build_report()
        report["pairs"][0]["missing"] = True

        assert "本轮没生成" in to_markdown(report)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
