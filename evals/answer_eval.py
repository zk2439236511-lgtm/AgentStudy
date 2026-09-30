"""答案层评测（补课路线 ⑤）：把线上那条 RAG 链路原样搬到 CMRC 金标集上，量三件事——
答得对不对（EM / char-F1）、有没有被检索到的上下文撑住（字面支撑率）、不该答时停没停（拒答混淆矩阵）。

三个刻意的设计决定：

1. 被测对象是 `create_rag_chain_with_sources` 本身，不是另写一份近似的链路——
   否则测出来的分数不能代表 kb-web 线上行为。
2. 两组条件对比：阈值开（d ≤ 1.0，检索层先拦一道）vs 阈值关（无关上下文照样塞给模型，
   只看提示词里那句 "say so" 自己兜不兜得住）。第二组只对全部构造负样本 + 抽样可回答题跑，
   因为要控制调用量；它回答的是"阈值到底有没有换来东西"。
3. 被拒答的题不进 EM/F1 的分母，但同时报"全分母、被拒按 0 计"的保守口径。只看前者会自欺：
   模型拒得越多，作答子集越小，平均分越好看。

用法：
    python evals/answer_eval.py --limit 6 --probe 4     # 冒烟，十几次真调用
    python evals/answer_eval.py                          # 全量

会真实调用百炼 qwen-plus 与 embedding。索引建在系统临时目录，绝不碰 apps/knowledge-rag/data/。
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
APP_DIR = REPO_ROOT / "apps" / "knowledge-rag"
DATASET_DIR = EVALS_DIR / "datasets" / "cmrc2018"

sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(APP_DIR / "src"))

from dotenv import load_dotenv

load_dotenv(APP_DIR / ".env")

from answer_metrics import (
    abstention_stats,
    answer_scores,
    char_f1,
    faithfulness,
    looks_like_refusal,
)
from chroma_threshold_eval import build_store
from cmrc_retrieval_eval import read_jsonl, summarize, to_documents
from ragdemo.rag_chain import DEFAULT_K, DEFAULT_MAX_DISTANCE, create_rag_chain_with_sources


def row_from_result(item: dict, result: dict) -> dict:
    """把一道题的链路输出整理成一行证据。

    检索层拒答时 answer 是我们自己写的文案，不是模型答案——EM/F1/Faithfulness 全部留 None，
    免得拿拒答文案去和金标 span 比出一个莫名其妙的分数。模型自拒只在没被阈值拦下时才判。
    """
    docs = result["source_documents"]
    distances = [doc.metadata.get("score") for doc in docs]
    store_refused = bool(result["refused"])
    answer = result["answer"]
    contexts = [doc.page_content for doc in docs]

    row = {
        "question_id": item["question_id"],
        "question": item["question"],
        "answerable": item["answerable"],
        "gold_answers": item["gold_answers"],
        "top1_distance": round(float(distances[0]), 4) if distances and distances[0] is not None else None,
        "store_refused": store_refused,
        "model_self_refused": False if store_refused else looks_like_refusal(answer),
        "predicted": answer,
        "em": None,
        "f1": None,
        "containment": None,
        "negative_f1": None,
        "faithfulness_unigram": None,
        "faithfulness_bigram": None,
    }
    if store_refused:
        return row

    faith = faithfulness(answer, contexts)
    row["faithfulness_unigram"] = faith["unigram"]
    row["faithfulness_bigram"] = faith["bigram"]
    if item["answerable"]:
        scores = answer_scores(answer, item["gold_answers"])
        row["em"], row["f1"], row["containment"] = (
            scores["em"], scores["f1"], scores["containment"]
        )
    else:
        # 负样本自带的 gold 来自被扣住的那一段：硬答却答得上，说明答案来自模型参数而不是检索
        row["negative_f1"] = round(
            max((char_f1(answer, gold) for gold in item["gold_answers"]), default=0.0), 4
        )
    return row


def run_condition(chain, items: list[dict]) -> tuple[list[dict], dict]:
    rows, started = [], time.time()
    for item in items:
        rows.append(row_from_result(item, chain.invoke(item["question"])))
    generations = sum(1 for row in rows if not row["store_refused"])
    cost = {
        "questions": len(rows),
        "llm_generations": generations,
        "retrievals": len(rows),
        "seconds": round(time.time() - started, 1),
    }
    return rows, cost


def probe_items(golden: list[dict], n_answerable: int) -> list[dict]:
    """阈值关那组只跑：全部构造负样本 + 按步长抽的可回答题（给 0 就一道可回答题都不抽）。"""
    negatives = [item for item in golden if not item["answerable"]]
    answerable = [item for item in golden if item["answerable"]]
    if n_answerable <= 0:
        return negatives
    if n_answerable < len(answerable):
        stride = len(answerable) / n_answerable
        answerable = [answerable[int(index * stride)] for index in range(n_answerable)]
    return negatives + answerable


def head_mixed(items: list[dict], n: int) -> list[dict]:
    """冒烟取样：留 3 道负样本，其余按原顺序取可回答题，保证拒答与作答两条路径都被走到。"""
    if n <= 0:
        return []
    negatives = [item for item in items if not item["answerable"]][:3]
    positives = [item for item in items if item["answerable"]][: max(n - len(negatives), 0)]
    return negatives + positives


def condition_report(rows: list[dict], cost: dict) -> dict:
    stats = abstention_stats(rows)
    answered_f1 = [row["f1"] for row in rows if row["f1"] is not None]
    stats["f1_distribution"] = summarize(answered_f1)
    stats["cost"] = cost
    return stats


def _negative_hard_answer_line(probe: dict) -> str:
    """负样本被硬答时的说明。一道都没硬答时不能照常输出"F1 0.0"，那会被读成"答得很差"。"""
    stats = probe["negative"]
    if not stats["answered"]:
        return (
            f"- 负样本全部被拦下或自拒（{stats['either_refused']}/{stats['n']}），没有硬答，"
            "下面的参数知识比对无数据\n"
        )
    return (
        f"- 负样本被硬答 {stats['answered']} 道，平均 char-F1（对着它自带的真实答案算）："
        f"{probe['negative_hard_answered_f1']}，其中 F1 ≥ 0.5 的 "
        f"{probe['negative_hard_answered_matches_gold']} 道——答上不代表链路对，那是模型参数里的知识\n"
    )


def fmt_group(group: dict) -> str:
    return (
        f"题数 {group['n']}｜阈值拦下 {group['store_refused']}｜模型自拒 {group['model_self_refused']}"
        f"｜任一层拒 {group['either_refused']}｜硬答 {group['answered']}"
        f"｜拒答率 {group['refusal_rate']}"
    )


def to_markdown(report: dict) -> str:
    thresholds = report["conditions"]
    on = thresholds["threshold_on"]
    out = [
        "## 1. 配置与调用量\n\n",
        f"- 模型 `{report['chat_model']}`，k={report['k']}，chunk_size={report['chunk_size']}，"
        f"拒答线 d ≤ {report['max_distance']}（口径见 2026-09-30-chroma-threshold.md）\n",
        f"- 阈值开：{json.dumps(on['cost'], ensure_ascii=False)}\n",
        f"- 阈值关（探针组）：{json.dumps(thresholds['no_threshold_probe']['cost'], ensure_ascii=False)}\n",
        "- 生产链路是 `temperature=1`，同配置重跑分数会抖，看量级别小数点较真\n",
        "\n## 2. 阈值开：拒答与答案质量（全量金标集）\n\n",
        f"- 可回答题：{fmt_group(on['answerable'])}\n",
        f"- 构造负样本：{fmt_group(on['negative'])}\n\n",
        "| 口径 | n | EM | span 命中 | char-F1 | 字面支撑 unigram | bigram |\n"
        "|---|---|---|---|---|---|---|\n",
        f"| 作答子集 | {on['quality_answered_subset']['n']} | {on['quality_answered_subset']['em']} "
        f"| {on['quality_answered_subset']['containment']} | {on['quality_answered_subset']['f1']} "
        f"| {on['quality_answered_subset']['faithfulness_unigram']} "
        f"| {on['quality_answered_subset']['faithfulness_bigram']} |\n",
        f"| 全分母（被拒按 0 计） | {on['quality_all_answerable_zero_refused']['n']} "
        f"| {on['quality_all_answerable_zero_refused']['em']} "
        f"| {on['quality_all_answerable_zero_refused']['containment']} "
        f"| {on['quality_all_answerable_zero_refused']['f1']} | — | — |\n",
        "\n> EM 与 span 命中要分开看：gold 是原文里的短 span，模型答成完整句子时 EM 必然为 0，"
        "这时「答案里有没有含着正确的那段」只能靠 span 命中来判。\n",
        # 分母比作答子集大：模型自拒的可回答题 f1 记 0（参与判分），只有阈值拦下的才是 None。
        # 这句要是写回"作答子集"，读的人拿 n 一对就会发现对不上。
        f"\n- 可回答题里有分数的那些 char-F1 分布（n={on['f1_distribution']['n']}"
        f" = 作答 {on['quality_answered_subset']['n']} + 模型自拒记 0 分的"
        f" {on['answerable']['model_self_refused']}）：`{json.dumps(on['f1_distribution'], ensure_ascii=False)}`\n",
        "\n## 3. 阈值关（探针组）：只看提示词那句 say so 兜不兜得住\n\n",
        f"- 可回答题：{fmt_group(thresholds['no_threshold_probe']['answerable'])}\n",
        f"- 构造负样本：{fmt_group(thresholds['no_threshold_probe']['negative'])}\n",
        _negative_hard_answer_line(thresholds["no_threshold_probe"]),
        "\n## 4. 肉眼复核证据\n\n",
        "### 4.1 负样本：阈值开 vs 阈值关，同一道题模型说了什么\n\n",
        "| question_id | top1 距离 | 阈值开 | 阈值关（模型原文） |\n|---|---|---|---|\n",
    ]

    off_rows = {row["question_id"]: row for row in report["rows"]["no_threshold_probe"]}
    for row in report["rows"]["threshold_on"]:
        if row["answerable"]:
            continue
        off = off_rows.get(row["question_id"], {})
        out.append(
            f"| {row['question_id']} | {row['top1_distance']} "
            f"| {'阈值拦下' if row['store_refused'] else '放行'} "
            f"| {_clip(off.get('predicted', ''))} |\n"
        )

    out.append("\n### 4.2 可回答题：最差与最好的作答（EM/F1 只看这里，拒答的题不在其中）\n\n")
    out.append(
        "| question_id | gold | predicted | EM | span 命中 | F1 | bigram 支撑 |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    answered = [row for row in report["rows"]["threshold_on"]
                if row["answerable"] and row["f1"] is not None]
    answered.sort(key=lambda row: row["f1"])
    showcase = answered[:8] if len(answered) <= 11 else answered[:8] + answered[-3:]
    for row in showcase:
        out.append(
            f"| {row['question_id']} | {_clip(' / '.join(row['gold_answers']), 40)} "
            f"| {_clip(row['predicted'], 40)} | {row['em']} | {row['containment']} | {row['f1']} "
            f"| {row['faithfulness_bigram']} |\n"
        )
    out.append("\n## 5. 结论与限制（手工追加）\n\n")
    return "".join(out)


def _clip(text, limit: int = 80) -> str:
    if not text:
        return ""
    one_line = str(text).replace("\n", " ").replace("|", "\\|").strip()
    return one_line[:limit] + ("…" if len(one_line) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="答案层评测（EM / char-F1 / Faithfulness / 拒答）")
    parser.add_argument("--chunk-size", type=int, default=0, help="0 表示一段一块")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    parser.add_argument("--limit", type=int, default=0, help="阈值开那组只跑前 N 题，冒烟用")
    parser.add_argument("--probe", type=int, default=20, help="阈值关组额外抽多少道可回答题")
    parser.add_argument("--keep", action="store_true", help="保留临时索引目录")
    args = parser.parse_args()

    corpus = read_jsonl(DATASET_DIR / "corpus.jsonl")
    golden = read_jsonl(DATASET_DIR / "golden.jsonl")
    chunks = to_documents(corpus, args.chunk_size)

    positives = [item for item in golden if item["answerable"]]
    negatives = [item for item in golden if not item["answerable"]]
    all_items = positives + negatives
    on_items = head_mixed(all_items, args.limit) if args.limit else all_items
    probe_pool = probe_items(golden, args.probe)
    off_items = head_mixed(probe_pool, args.limit + args.probe) if args.limit else probe_pool

    persist_dir = tempfile.mkdtemp(prefix="cmrc-answer-eval-")
    print(f"建索引：{len(chunks)} 块 → {persist_dir}")
    started = time.time()
    store = build_store(chunks, persist_dir)
    print(f"建库 {round(time.time() - started, 2)}s")

    print(f"预计真调用：embedding 查询 {len(on_items) + len(off_items)} 次，"
          f"生成最多 {len(on_items) + len(off_items)} 次（阈值拦下的不发生成）")

    on_chain = create_rag_chain_with_sources(
        store, k=args.k, max_distance=args.max_distance
    )
    on_rows, on_cost = run_condition(on_chain, on_items)
    print(f"阈值开完成：{json.dumps(on_cost, ensure_ascii=False)}")

    # 阈值关：同一批题把无关上下文照样塞给模型，看提示词层的出口够不够
    off_chain = create_rag_chain_with_sources(store, k=args.k, max_distance=None)
    off_rows, off_cost = run_condition(off_chain, off_items)
    print(f"阈值关完成：{json.dumps(off_cost, ensure_ascii=False)}")

    report = {
        "date": str(date.today()),
        "chat_model": os.getenv("RAG_CHAT_MODEL", "qwen-plus"),
        "k": args.k,
        "chunk_size": args.chunk_size,
        "max_distance": args.max_distance,
        "corpus_docs": len(corpus),
        "chunks": len(chunks),
        "conditions": {
            "threshold_on": condition_report(on_rows, on_cost),
            "no_threshold_probe": condition_report(off_rows, off_cost),
        },
        "rows": {"threshold_on": on_rows, "no_threshold_probe": off_rows},
    }

    results_dir = EVALS_DIR / "results"
    stem = f"{report['date']}-answer"
    (results_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / f"{stem}.md").write_text(to_markdown(report), encoding="utf-8")
    print(f"结果：evals/results/{stem}.{{json,md}}")

    if args.keep:
        print(f"索引保留在 {persist_dir}")
    else:
        shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
