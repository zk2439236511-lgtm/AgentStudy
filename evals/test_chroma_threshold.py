"""chroma_threshold_eval.py 里那三个纯函数的离线用例。

建库、检索都要打百炼，这里只测"读结果、算对齐、渲染报告"这几步——
它们出错的话，真实跑出来的数字一样会骗人。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "knowledge-rag" / "src"))

from chroma_threshold_eval import (
    cross_check_with_memory,
    pick_calibration_sample,
    to_markdown,
)


def test_pick_calibration_sample_takes_every_negative_first():
    """负样本是阈值线另一侧的证据，名额不够时也要全留；剩下的均匀铺在可回答题上。"""
    golden = [
        {"question_id": f"N{i}", "answerable": False} for i in range(3)
    ] + [{"question_id": f"A{i}", "answerable": True} for i in range(10)]

    sample = pick_calibration_sample(golden, 8)

    assert [item["question_id"] for item in sample[:3]] == ["N0", "N1", "N2"]
    assert len(sample) == 8
    # 可回答题按步长取样（10 道取 5 道，步长 2），不会全挤在开头
    assert [item["question_id"] for item in sample[3:]] == ["A0", "A2", "A4", "A6", "A8"]


def test_pick_calibration_sample_when_negatives_are_more_than_the_budget():
    golden = [{"question_id": f"N{i}", "answerable": False} for i in range(5)]

    assert len(pick_calibration_sample(golden, 2)) == 5


def make_previous(path: Path, chunk_size: int, k: int, per_question: list[dict]) -> None:
    path.write_text(
        json.dumps([{"chunk_size": chunk_size, "k": k, "per_question": per_question}]),
        encoding="utf-8",
    )


def test_cross_check_confirms_the_conversion(tmp_path):
    """换算对了，误差就该是 0：d=1.2（l2 口径）对 cos=0.4，d=1.4 对 cos=0.3。"""
    previous = tmp_path / "previous.json"
    make_previous(
        previous,
        0,
        8,
        [
            {"question_id": "Q1", "answerable": True, "top1_similarity": 0.4, "gold_rank": 1},
            {"question_id": "Q2", "answerable": False, "top1_similarity": 0.3},
        ],
    )
    rows = [
        {"question_id": "Q1", "answerable": True, "top1_distance": 1.2, "gold_rank": 1},
        {"question_id": "Q2", "answerable": False, "top1_distance": 1.4},
    ]

    result = cross_check_with_memory(rows, previous, "l2", 0, 8)

    assert result["compared"] == 2
    assert result["gold_rank_differences"] == "0/1"
    assert result["converted_cosine_error"] == {"mean": 0.0, "max": 0.0}


def test_cross_check_exposes_a_wrong_space_assumption(tmp_path):
    """口径说错时误差会顶到 0.6 这种量级——这正是要它替我们把关的地方。"""
    previous = tmp_path / "previous.json"
    make_previous(
        previous,
        0,
        8,
        [{"question_id": "Q1", "answerable": True, "top1_similarity": 0.4, "gold_rank": 1}],
    )
    rows = [{"question_id": "Q1", "answerable": True, "top1_distance": 1.2, "gold_rank": 1}]

    assert cross_check_with_memory(rows, previous, "cosine", 0, 8)["converted_cosine_error"][
        "max"
    ] == 0.6


def test_cross_check_reports_the_configured_pair_is_missing(tmp_path):
    previous = tmp_path / "previous.json"
    make_previous(previous, 500, 4, [])

    assert "skipped" in cross_check_with_memory([], previous, "l2", 0, 8)


def test_cross_check_when_previous_file_is_absent(tmp_path):
    missing = tmp_path / "nope.json"

    assert "skipped" in cross_check_with_memory([], missing, "l2", 0, 8)


def minimal_report(**overrides) -> dict:
    report = {
        "chunk_size": 0,
        "k": 8,
        "chunks": 100,
        "build_seconds": 5.0,
        "hit_at_k": 1.0,
        "answer_in_context": 1.0,
        "space_evidence": {"configuration": {"hnsw": {"space": "l2"}}, "metadata": None},
        "calibration": {
            "n_pairs": 2,
            "corpus_norm": {"min": 1.0, "max": 1.0, "mean": 1.0},
            "query_norm": {"min": 1.0, "max": 1.0},
            "same_top_doc_count": "2/2",
            "verdict": {
                "best": "l2",
                "confirmed": True,
                "n_pairs": 2,
                "ranking": [
                    {"space": "l2", "max_abs_error": 0.0, "mean_abs_error": 0.0},
                    {"space": "cosine", "max_abs_error": 0.6, "mean_abs_error": 0.3},
                    {"space": "ip", "max_abs_error": 2.4, "mean_abs_error": 1.2},
                ],
            },
            "pairs": [],
        },
        "answerable_top1_distance": {
            "n": 2, "min": 0.4, "p25": 0.4, "median": 0.6, "p75": 0.8, "max": 0.8, "mean": 0.6,
        },
        "negative_top1_distance": {
            "n": 2, "min": 1.2, "p25": 1.2, "median": 1.3, "p75": 1.4, "max": 1.4, "mean": 1.3,
        },
        "distance_threshold_sweep": [
            {
                "max_distance": 1.0,
                "negative_refusal_rate": 1.0,
                "negative_refused": "2/2",
                "answerable_false_refusal_rate": 0.0,
                "answerable_wrongly_refused": "0/2",
                "equivalent_cosine": 0.5,
            }
        ],
        "cross_check": {"compared": 2, "gold_rank_differences": "0/2",
                        "converted_cosine_error": {"mean": 0.0, "max": 0.001}},
    }
    report.update(overrides)
    return report


def test_to_markdown_shows_the_confirmed_space_and_operating_point():
    rendered = to_markdown(minimal_report())

    assert "判定的口径：**l2**" in rendered
    assert "等价余弦线（口径 l2）" in rendered
    assert "| 1.0 | 0.5 | 1.0 | 2/2 | 0.0 | 0/2 |" in rendered


def test_to_markdown_survives_a_run_without_negatives():
    """冒烟时只跑了前几道题、一组都没有负样本——报告不能因此崩掉。"""
    report = minimal_report(
        negative_top1_distance={"n": 0},
        distance_threshold_sweep=[],
    )

    rendered = to_markdown(report)

    assert "构造负样本 | 0 | 这组没有题，跳过" in rendered
    assert "跳过阈值扫描" in rendered
    assert "## 4." in rendered
