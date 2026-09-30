"""提示词单变量 A/B：把 ⑤ 定位出来的"答案啰嗦"真的改一次，再按同一套指标重测。

定位来源（`evals/results/2026-09-30-answer.md`）：作答子集 span 命中 0.9077 而 char-F1 只有
0.5163 —— 正确的片段在答案里，只是被整句复述冲淡了。按外部评审的对照表这属于
`Answer Relevancy / 输出格式` 那一栏，该改的是提示词，不是检索。

设计要点：

1. **只有一个变量**。两组都用线上链 `create_rag_chain_with_sources`、同一个临时 Chroma、
   同 k=8、同拒答线 d ≤ 1.0、同模型同 temperature，只有提示词模板不同。
2. **判据先写死再跑**（下面 `PASS_*` 那四个常量）。事后按结果找解释等于没做实验。
3. **配对而不是分组比较**。12 道题在两组各跑一次，逐题算差值——小样本里"两组的平均值"
   会被题目难度差异淹掉，配对能把它消掉。
4. **对照组是重跑的，不是引用昨天的记录**。同一天同一套索引重跑基线，才知道
   `temperature=1` 本身的抖动有多大；这个抖动以后就是所有小样本实验的噪声地板。
5. **默认不花钱**。不带 `--yes` 只打印选题和预算，带 `--yes` 才真调百炼。

用法：
    python evals/prompt_ab_eval.py                    # 干跑：看选了哪些题、要花多少 token
    python evals/prompt_ab_eval.py --yes --out-suffix ab12   # 实跑
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
APP_DIR = REPO_ROOT / "apps" / "knowledge-rag"
DATASET_DIR = EVALS_DIR / "datasets" / "cmrc2018"
BASELINE_JSON = EVALS_DIR / "results" / "2026-09-30-answer.json"

sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(APP_DIR / "src"))

from dotenv import load_dotenv

load_dotenv(APP_DIR / ".env")

from answer_eval import condition_report, fmt_cost, row_from_result
from chroma_threshold_eval import build_store
from cmrc_retrieval_eval import read_jsonl, summarize, to_documents
from ragdemo.rag_chain import DEFAULT_K, DEFAULT_MAX_DISTANCE, RAG_TEMPLATE, create_rag_chain_with_sources

# 与基线只差这一小段：前两句原样保留，免得把"允许拒答"这个 affordance 一起改掉
CONCISE_TEMPLATE = """Answer the question based only on the following context.
If the context doesn't contain enough information to answer, say so.

Answer as briefly as possible: output the shortest span of text from the context that
answers the question. Do not restate the question and do not explain.

Context:
{context}

Question: {question}

Answer:"""

# 判据：跑之前定的，不看结果改
PASS_F1_GAIN = 0.05        # 平均 char-F1 至少涨这么多
PASS_CONTAINMENT_DROP = 0.05  # span 命中不能跌超过这么多
PASS_SENTINEL_FLOOR = 0.5  # 满分组（基线 F1=1.0）跌到这条线以下算回归
PASS_SENTINEL_TOLERANCE = 1   # 满分组最多允许 1 道回归

# 单价：2026-09-30 两次真机冒烟分别测到 2611.7 和 3027.0 input/生成，取偏高那个做预算
# —— 预算估低了会直接撞额度上限，估高了只是提前知道要花多少
MEAN_INPUT_TOKENS_PER_GENERATION = 3000


def answered_rows(path: Path = BASELINE_JSON) -> list[dict]:
    """基线里"可回答且真发生了生成"的行——被阈值拦下的题 f1 是 None，不参与选题。"""
    report = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in report["rows"]["threshold_on"]
            if row["answerable"] and row["f1"] is not None]


def select_strata(rows: list[dict], n_low: int = 5, n_mid: int = 4, n_high: int = 3) -> list[dict]:
    """分层选配对题，三档各有用途：

    - low：F1 < 0.3，其中多数 span 都没命中，是"真答错/答飞了"那一头；
    - mid：0.4 ≤ F1 ≤ 0.6 且答案最长，是本轮改动**主要**该救的那批——答对了但被复述冲淡；
    - high：基线 F1 = 1.0 的最短答案，本来就满分，只可能变坏，当回归哨兵用。

    返回顺序固定 low → mid → high，且带上 `stratum`，冒烟时按 head 取也能覆盖三档。
    """
    low = sorted((r for r in rows if r["f1"] < 0.3), key=lambda r: (r["f1"], -len(r["predicted"])))
    mid_pool = [r for r in rows if 0.4 <= r["f1"] <= 0.6]
    mid = sorted(mid_pool, key=lambda r: -len(r["predicted"]))
    high = sorted((r for r in rows if r["f1"] >= 0.999), key=lambda r: len(r["predicted"]))

    picked, seen = [], set()
    for stratum, pool, count in (("low", low, n_low), ("mid", mid, n_mid), ("high", high, n_high)):
        taken = 0
        for row in pool:
            if taken >= count:
                break
            if row["question_id"] in seen:
                continue
            seen.add(row["question_id"])
            picked.append({**row, "stratum": stratum})
            taken += 1
    return picked


def items_for(golden: list[dict], picked: list[dict]) -> list[dict]:
    """按选题顺序取金标集原题，保证喂给链路的 question/gold 和基线是同一份。"""
    by_id = {item["question_id"]: item for item in golden}
    return [by_id[row["question_id"]] for row in picked if row["question_id"] in by_id]


def quality(rows: list[dict]) -> dict:
    """只统计真发生了生成的题（f1 不为 None）。"""
    scored = [row for row in rows if row["f1"] is not None]
    return {
        "n": len(scored),
        "em": round(statistics.mean(row["em"] for row in scored), 4) if scored else None,
        "containment": round(statistics.mean(row["containment"] for row in scored), 4) if scored else None,
        "f1": round(statistics.mean(row["f1"] for row in scored), 4) if scored else None,
        "bigram": round(statistics.mean(row["faithfulness_bigram"] for row in scored), 4) if scored else None,
        "mean_answer_chars": round(statistics.mean(len(row["predicted"]) for row in scored), 1) if scored else None,
        "mean_output_tokens": round(statistics.mean(row["output_tokens"] for row in scored), 1) if scored else None,
    }


def paired(base: list[dict], variant: list[dict]) -> list[dict]:
    """逐题配对。基线有分、这轮却没生成出来（被拦下或链路异常）的行要留痕，不能悄悄丢。"""
    variant_by_id = {row["question_id"]: row for row in variant}
    lines = []
    for row in base:
        new = variant_by_id.get(row["question_id"])
        if new is None:
            continue
        lines.append({
            "question_id": row["question_id"],
            "stratum": row["stratum"],
            "f1_before": row["f1"],
            "f1_after": new["f1"],
            "containment_before": row["containment"],
            "containment_after": new["containment"],
            "chars_before": len(row["predicted"]),
            "chars_after": len(new["predicted"]) if new["f1"] is not None else None,
            "predicted_after": new["predicted"],
            "gold": row["gold_answers"],
            "missing": new["f1"] is None,
        })
    return lines


def decide(base: list[dict], variant: list[dict], pairs: list[dict]) -> list[dict]:
    """把 PASS_* 四条判据逐条判一遍。每条都要能指着自己的数字说明为什么成/不成。"""
    before, after = quality(base), quality(variant)
    verdicts = []

    f1_gain = None if before["f1"] is None or after["f1"] is None else round(after["f1"] - before["f1"], 4)
    verdicts.append({
        "name": f"主判据：平均 char-F1 涨 ≥ {PASS_F1_GAIN}",
        "value": f"{before['f1']} → {after['f1']}（Δ {f1_gain}）",
        "passed": f1_gain is not None and f1_gain >= PASS_F1_GAIN,
    })

    drop = None if before["containment"] is None or after["containment"] is None else round(
        before["containment"] - after["containment"], 4
    )
    verdicts.append({
        "name": f"护栏：span 命中跌幅 ≤ {PASS_CONTAINMENT_DROP}",
        "value": f"{before['containment']} → {after['containment']}（跌 {drop}）",
        "passed": drop is not None and drop <= PASS_CONTAINMENT_DROP,
    })

    sentinel = [p for p in pairs if p["stratum"] == "high"]
    regressed = [p for p in sentinel if p["missing"] or (p["f1_after"] or 0) < PASS_SENTINEL_FLOOR]
    verdicts.append({
        "name": f"哨兵：满分组跌破 {PASS_SENTINEL_FLOOR} 的不超过 {PASS_SENTINEL_TOLERANCE} 道",
        "value": f"回归 {len(regressed)}/{len(sentinel)} 道"
                 + (f"（{', '.join(p['question_id'] for p in regressed)}）" if regressed else ""),
        "passed": len(regressed) <= PASS_SENTINEL_TOLERANCE,
    })

    length_drop = None if before["mean_answer_chars"] is None or after["mean_answer_chars"] is None else round(
        before["mean_answer_chars"] - after["mean_answer_chars"], 1
    )
    verdicts.append({
        "name": "机制：平均答案字数确实下降（不降就说明涨分不是这段话带来的）",
        "value": f"{before['mean_answer_chars']} → {after['mean_answer_chars']} 字（降 {length_drop}）",
        "passed": length_drop is not None and length_drop > 0,
    })

    return [{"before": before, "after": after, "all_passed": all(v["passed"] for v in verdicts)}, verdicts]


def budget(items: list[dict]) -> dict:
    generations = len(items) * 2
    return {
        "questions": len(items),
        "generations": generations,
        "estimated_input_tokens": generations * MEAN_INPUT_TOKENS_PER_GENERATION,
        "note": f"单价按真机冒烟取 {MEAN_INPUT_TOKENS_PER_GENERATION} input/生成；两组都跑所以 ×2",
    }


def to_markdown(report: dict) -> str:
    head = [
        "## 1. 实验设计（判据先写死，再看结果）\n\n",
        f"- 变量：提示词模板。基线 `RAG_TEMPLATE` vs 实验组 `CONCISE_TEMPLATE`（只多一段「最短片段、别复述」）\n",
        f"- 固定项：模型 `{report['chat_model']}`、k={report['k']}、chunk_size={report['chunk_size']}、"
        f"拒答线 d ≤ {report['max_distance']}、同一临时 Chroma、temperature=1\n",
        f"- 选题：从基线 `{report['baseline'].name}` 分层配对 {report['budget']['questions']} 道"
        f"（档位 {'/'.join(report['strata'])}），两组各跑一遍 → 共 {report['budget']['generations']} 次生成\n",
        f"- 预算：{json.dumps(report['budget'], ensure_ascii=False)}\n",
        "\n## 2. 判据与结论\n\n",
        "| 判据 | 数字 | 是否达成 |\n|---|---|---|\n",
    ]
    for verdict in report["verdicts"]:
        head.append(f"| {verdict['name']} | {verdict['value']} | {'✅' if verdict['passed'] else '❌'} |\n")

    summary = report["summary"]
    head.append(
        f"\n- 一句话结论（写代码时就定好的规则，不靠肉眼）：**"
        f"{'四条全过 → 改动成立，可以进默认提示词' if summary['all_passed'] else '未全过 → 改动不成立或证据不足，不进生产默认值'}**\n"
    )

    head.append("\n## 3. 两组整体数字\n\n")
    head.append("| 指标 | 基线模板 | 最短片段模板 | Δ |\n|---|---|---|---|\n")
    before, after = summary["before"], summary["after"]
    for key, label in (("em", "EM"), ("containment", "span 命中"), ("f1", "char-F1"),
                       ("bigram", "字面支撑 bigram"), ("mean_answer_chars", "平均答案字数"),
                       ("mean_output_tokens", "平均输出 token"), ("n", "有分题数")):
        b, a = before[key], after[key]
        delta = "—" if b is None or a is None else round(a - b, 4)
        head.append(f"| {label} | {b} | {a} | {delta} |\n")

    head.append("\n## 4. 逐题配对（这才是判分的证据）\n\n")
    head.append("| question_id | 档 | F1 前→后 | span 前→后 | 字数 前→后 | 改后模型原文 |\n"
                "|---|---|---|---|---|---|\n")
    for line in report["pairs"]:
        if line["missing"]:
            head.append(f"| {line['question_id']} | {line['stratum']} | {line['f1_before']} → 本轮没生成 | "
                        f"{line['containment_before']} → — | {line['chars_before']} → — | 需查拒答 |\n")
            continue
        head.append(
            f"| {line['question_id']} | {line['stratum']} | {line['f1_before']} → {line['f1_after']} | "
            f"{line['containment_before']} → {line['containment_after']} | "
            f"{line['chars_before']} → {line['chars_after']} | {_clip(line['predicted_after'], 50)} |\n"
        )

    head.append("\n## 5. 成本\n\n")
    head.append(f"- 基线组：{fmt_cost(report['cost']['baseline'])}\n")
    head.append(f"- 实验组：{fmt_cost(report['cost']['variant'])}\n")
    head.append("\n## 6. 限制（手工追加结论时逐条对照）\n\n")
    head.append(f"- n={report['budget']['questions']}，只够看方向，不够判显著；`temperature=1` 的抖动量级"
                "正好用「基线重跑 vs 昨天的基线记录」来估\n")
    head.append("- char-F1 奖励抄写、惩罚改写：更短的答案天然更容易抬高它，所以必须和 span 命中、"
                "字数三条一起看，单看 F1 会被自己骗到\n")
    head.append("- 答案变短后 `looks_like_refusal` 的判定口径可能松动（短「不知道」更容易命中弱标记）\n")
    head.append("- 本轮不动检索，所以它只回答「提示词层还有多少油水」，不回答「检索层是不是最优」\n")
    return "".join(head)


def _clip(text, limit: int = 80) -> str:
    if not text:
        return ""
    one_line = str(text).replace("\n", " ").replace("|", "\\|").strip()
    return one_line[:limit] + ("…" if len(one_line) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="提示词单变量 A/B（最短片段 vs 现模板）")
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    parser.add_argument("--n-low", type=int, default=5)
    parser.add_argument("--n-mid", type=int, default=4)
    parser.add_argument("--n-high", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 道（冒烟用，会保持三档混合）")
    parser.add_argument("--yes", action="store_true", help="确认要真调百炼，否则只干跑打印预算")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--out-suffix", default="ab")
    args = parser.parse_args()

    picked = select_strata(answered_rows(), args.n_low, args.n_mid, args.n_high)
    if args.limit:
        picked = picked[: args.limit]
    golden = read_jsonl(DATASET_DIR / "golden.jsonl")
    items = items_for(golden, picked)
    plan = budget(items)

    print(f"选题 {len(items)} 道：" + ", ".join(f"{r['question_id']}({r['stratum']})" for r in picked))
    print(f"预算：{plan['generations']} 次生成 ≈ {plan['estimated_input_tokens']} input token"
          f"（{plan['note']}）")
    if not args.yes:
        print("干跑结束：加 --yes 才会真调百炼")
        return

    corpus = read_jsonl(DATASET_DIR / "corpus.jsonl")
    chunks = to_documents(corpus, args.chunk_size)
    persist_dir = tempfile.mkdtemp(prefix="cmrc-prompt-ab-")
    print(f"建索引：{len(chunks)} 块 → {persist_dir}")
    started = time.time()
    store = build_store(chunks, persist_dir)
    print(f"建库 {round(time.time() - started, 2)}s")

    # 对照组重跑（不是引用昨天的记录）：同一天同一套索引，才能把抖动和提示词的影响分开
    baseline_chain = create_rag_chain_with_sources(store, k=args.k, max_distance=args.max_distance)
    variant_chain = create_rag_chain_with_sources(
        store, k=args.k, max_distance=args.max_distance, template=CONCISE_TEMPLATE
    )

    from answer_eval import run_condition

    base_rows, base_cost = run_condition(baseline_chain, items)
    print(f"基线模板完成：{fmt_cost(base_cost)}")
    variant_rows, variant_cost = run_condition(variant_chain, items)
    print(f"最短片段模板完成：{fmt_cost(variant_cost)}")

    # 配对用的"基线"取本轮重跑的结果，昨天的记录只用来选档
    base_for_pairs = [{**row, "stratum": picked[index]["stratum"]}
                      for index, row in enumerate(base_rows)]
    pairs = paired(base_for_pairs, variant_rows)
    summary, verdicts = decide(base_for_pairs, variant_rows, pairs)

    report = {
        "date": str(date.today()),
        "chat_model": os.getenv("RAG_CHAT_MODEL", "qwen-plus"),
        "k": args.k,
        "chunk_size": args.chunk_size,
        "max_distance": args.max_distance,
        "baseline": BASELINE_JSON,
        "budget": plan,
        "selection": [{"question_id": r["question_id"], "stratum": r["stratum"], "f1_baseline_record": r["f1"]}
                      for r in picked],
        "strata": sorted({r["stratum"] for r in picked}),
        "conditions": {
            "baseline_template": condition_report(base_rows, base_cost),
            "concise_template": condition_report(variant_rows, variant_cost),
        },
        "summary": summary,
        "verdicts": verdicts,
        "pairs": pairs,
        "cost": {"baseline": base_cost, "variant": variant_cost},
        "rows": {"baseline": base_rows, "variant": variant_rows},
    }

    stem = f"{report['date']}-prompt-ab-{args.out_suffix}"
    results_dir = EVALS_DIR / "results"
    (results_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (results_dir / f"{stem}.md").write_text(to_markdown(report), encoding="utf-8")
    print(f"结果：evals/results/{stem}.{{json,md}}")
    for verdict in verdicts:
        print(f"{'达成' if verdict['passed'] else '未达成'}｜{verdict['name']}｜{verdict['value']}")

    shutil.rmtree(persist_dir, ignore_errors=True) if not args.keep else print(f"索引保留在 {persist_dir}")


if __name__ == "__main__":
    main()
