"""线上 Chroma 口径下的阈值拒答实验（补课路线 ④）。

上一轮 evals/cmrc_retrieval_eval.py 给的是内存库的**余弦相似度**分布（越大越相关）；
线上 kb-web 走 Chroma，透出的是**距离**（越小越相关）。上次的 0.5~0.6 那根线不能直接搬，
所以这个脚本把同一份金标集、同一套切块搬到 Chroma 上重跑一遍，回答三个问题：

1. 这张索引实际是哪种距离口径——用「库返回的距离」和「手工算的余弦」实测配对反推，不靠猜；
2. 距离口径下，可回答题与构造负样本的 top1 距离分别落在哪；
3. 「top1 距离大于 t 就拒答」在各个 t 上的正确拒答率与误伤率。

用法：
    python evals/chroma_threshold_eval.py --limit 20 --calibrate 8   # 冒烟（少量真调用）
    python evals/chroma_threshold_eval.py                              # 全量

会真实调用百炼 embedding。索引建在系统临时目录，绝不碰 apps/knowledge-rag/data/，
也不会把 CMRC 语料混进线上知识库。
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
APP_DIR = REPO_ROOT / "apps" / "knowledge-rag"
DATASET_DIR = EVALS_DIR / "datasets" / "cmrc2018"

sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(APP_DIR / "src"))

from dotenv import load_dotenv

load_dotenv(APP_DIR / ".env")

from cmrc_retrieval_eval import read_jsonl, summarize, to_documents
from metrics import (
    cosine_from_distance,
    distance_refusal_sweep,
    match_distance_space,
)
from ragdemo.ingest import load_vector_store
from ragdemo.vector_store import EMBEDDING_BATCH_SIZE, create_embeddings


def build_store(chunks: list, persist_dir: str):
    """用线上的建库代码路径，只是把目录指到别处。"""
    store = load_vector_store(persist_dir, create_embeddings())
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        store.add_documents(chunks[start : start + EMBEDDING_BATCH_SIZE])
    return store


def read_space_evidence(store) -> dict:
    """原样读出 collection 的距离配置，作为"口径是什么"的第一手证据。"""
    collection = store._collection
    evidence = {}
    for attribute in ("configuration", "metadata"):
        value = getattr(collection, attribute, None)
        evidence[attribute] = json.loads(json.dumps(value, default=str)) if value else None
    return evidence


def corpus_vectors(store) -> tuple[list[str], np.ndarray, np.ndarray]:
    """把已写盘的语料向量读回来（不再付一次 embedding）。

    chromadb 的 get() 默认不带 metadatas/embeddings，必须写进 include；
    这里要的是"向量 + 它属于哪一段"的对应关系。
    """
    fetched = store._collection.get(include=["embeddings", "metadatas"])
    metadatas = fetched["metadatas"] or [{} for _ in fetched["ids"]]
    doc_ids = [(meta or {}).get("doc_id") for meta in metadatas]
    matrix = np.asarray(fetched["embeddings"], dtype=float)
    return doc_ids, matrix, np.linalg.norm(matrix, axis=1)


def pick_calibration_sample(golden: list[dict], n: int) -> list[dict]:
    """负样本全取（它们是阈值线另一侧的证据），剩下按步长均匀取可回答题。"""
    negatives = [item for item in golden if not item["answerable"]]
    answerable = [item for item in golden if item["answerable"]]
    remaining = max(n - len(negatives), 0)
    if 0 < remaining < len(answerable):
        stride = len(answerable) / remaining
        answerable = [answerable[int(index * stride)] for index in range(remaining)]
    return negatives + answerable


def calibrate_distance_space(store, embeddings, golden: list[dict], n: int) -> dict:
    """实测「库返回距离 vs 手工余弦」的配对，反推这张库是哪种距离口径。

    手工余弦是拿同一个 embed_query 向量和盘上的语料向量直接算的，所以只要向量归一化过、
    且库的距离是余弦的单调函数，手工余弦的 argmax 就该和库的 top1 是同一段。
    """
    doc_ids, matrix, norms = corpus_vectors(store)
    sample = pick_calibration_sample(golden, n)

    pairs, rows, query_norms = [], [], []
    for item in sample:
        query_vector = np.asarray(
            embeddings.embed_query(item["question"]), dtype=float
        )
        query_norms.append(float(np.linalg.norm(query_vector)))
        similarities = (matrix @ query_vector) / (norms * np.linalg.norm(query_vector))
        best = int(np.argmax(similarities))

        hits = store.similarity_search_with_score(item["question"], k=1)
        distance = float(hits[0][1]) if hits else float("nan")
        library_doc = hits[0][0].metadata.get("doc_id") if hits else None
        pairs.append((distance, float(similarities[best])))
        rows.append(
            {
                "question_id": item["question_id"],
                "answerable": item["answerable"],
                "library_distance": round(distance, 6),
                "manual_cosine": round(float(similarities[best]), 6),
                "manual_top_doc": doc_ids[best],
                "library_top_doc": library_doc,
                "same_top_doc": library_doc == doc_ids[best],
            }
        )

    verdict = match_distance_space(pairs)
    return {
        "n_pairs": len(pairs),
        "corpus_norm": {
            "min": round(float(norms.min()), 6),
            "max": round(float(norms.max()), 6),
            "mean": round(float(norms.mean()), 6),
        },
        "query_norm": {
            "min": round(min(query_norms), 6),
            "max": round(max(query_norms), 6),
        },
        "same_top_doc_count": f"{sum(1 for r in rows if r['same_top_doc'])}/{len(rows)}",
        "verdict": verdict,
        "pairs": rows,
    }


def probe(store, golden: list[dict], k: int) -> list[dict]:
    """逐题检索一次，记下 top1 距离和标准答案段落在结果里的名次（从 1 起，None=没进前 K）。"""
    rows = []
    for item in golden:
        hits = store.similarity_search_with_score(item["question"], k=k)
        doc_ids = [doc.metadata.get("doc_id") for doc, _ in hits]
        distances = [round(float(score), 4) for _, score in hits]
        gold = set(item["gold_doc_ids"])
        rank = next((i + 1 for i, d in enumerate(doc_ids) if d in gold), None)
        row = {
            "question_id": item["question_id"],
            "answerable": item["answerable"],
            "top1_distance": distances[0] if distances else None,
            "top1_doc_id": doc_ids[0] if doc_ids else None,
            "distances": distances,
        }
        if item["answerable"]:
            row.update(
                {
                    "gold_rank": rank,
                    "hit": 1.0 if rank else 0.0,
                    "answer_in_context": 1.0
                    if any(
                        answer in "\n".join(doc.page_content for doc, _ in hits)
                        for answer in item["gold_answers"]
                    )
                    else 0.0,
                }
            )
        rows.append(row)
    return rows


def cross_check_with_memory(rows: list[dict], previous_path: Path, space: str,
                            chunk_size: int, k: int) -> dict:
    """和上一轮内存库的结果逐题对齐：换算后的余弦应该等于当时实测的余弦相似度。

    这一步才是"阈值可以搬"的依据：名次一致 + 分数对得上，才敢把 ④ 的工作点写进线上代码。
    """
    if not previous_path.exists():
        return {"skipped": f"找不到 {previous_path.name}，跳过对齐"}

    previous = next(
        (
            row
            for row in json.loads(previous_path.read_text(encoding="utf-8"))
            if row["chunk_size"] == chunk_size and row["k"] == k
        ),
        None,
    )
    if previous is None:
        return {"skipped": f"{previous_path.name} 里没有 chunk_size={chunk_size}, k={k} 这组"}

    by_id = {item["question_id"]: item for item in previous["per_question"]}
    rank_diffs, score_errors, compared = [], [], 0
    for row in rows:
        old = by_id.get(row["question_id"])
        if not old or old.get("top1_similarity") is None or row["top1_distance"] is None:
            continue
        compared += 1
        if row["answerable"]:
            rank_diffs.append(1 if old.get("gold_rank") != row["gold_rank"] else 0)
        converted = cosine_from_distance(row["top1_distance"], space)
        score_errors.append(abs(converted - float(old["top1_similarity"])))

    return {
        "previous_file": previous_path.name,
        "compared": compared,
        "gold_rank_differences": f"{sum(rank_diffs)}/{len(rank_diffs)}" if rank_diffs else "n/a",
        "converted_cosine_error": {
            "mean": round(sum(score_errors) / len(score_errors), 4) if score_errors else None,
            "max": round(max(score_errors), 4) if score_errors else None,
        },
    }


def to_markdown(report: dict) -> str:
    verdict = report["calibration"]["verdict"]
    out = [
        "## 1. 距离口径证据（实测反推，不是假设）\n\n",
        f"- collection 配置：`{json.dumps(report['space_evidence'], ensure_ascii=False)}`\n",
        f"- 语料向量模长：{report['calibration']['corpus_norm']}；"
        f"问题向量模长：{report['calibration']['query_norm']}\n",
        f"- 库的 top1 与手工余弦的 top1 是同一段：{report['calibration']['same_top_doc_count']}\n",
        f"- 实测 {verdict['n_pairs']} 对，判定的口径：**{verdict['best']}**"
        f"（confirmed={verdict['confirmed']}）\n\n",
        "| 候选口径 | 最大残差 | 平均残差 |\n|---|---|---|\n",
    ]
    for row in verdict["ranking"]:
        out.append(f"| {row['space']} | {row['max_abs_error']} | {row['mean_abs_error']} |\n")

    out.append(
        "\n## 2. top1 距离分布（越小越相关，线上 Chroma 口径）\n\n"
        "| 组 | n | min | p25 | 中位 | p75 | max | mean |\n|---|---|---|---|---|---|---|---|\n"
    )
    for key, label in (
        ("answerable_top1_distance", "可回答题"),
        ("negative_top1_distance", "构造负样本"),
    ):
        stats = report[key]
        if not stats.get("n"):
            out.append(f"| {label} | 0 | 这组没有题，跳过 |\n")
            continue
        out.append(
            f"| {label} | {stats['n']} | {stats['min']} | {stats['p25']} | {stats['median']} "
            f"| {stats['p75']} | {stats['max']} | {stats['mean']} |\n"
        )

    if not report["distance_threshold_sweep"]:
        out.append("\n## 3. 阈值扫描\n\n（这批题里没有构造负样本，跳过阈值扫描）\n")
    else:
        converted = any("equivalent_cosine" in row for row in report["distance_threshold_sweep"])
        column = (
            f"等价余弦线（口径 {verdict['best']}）" if converted else "等价余弦线（口径未确认）"
        )
        out.append(
            "\n## 3. 阈值扫描：把「top1 距离大于 t 就拒答」当成分类器\n\n"
            f"| max_distance | {column} | 负样本正确拒答 | 负样本 "
            "| 可回答题误伤 | 可回答题 |\n|---|---|---|---|---|---|\n"
        )
        for row in report["distance_threshold_sweep"]:
            out.append(
                f"| {row['max_distance']} | {row.get('equivalent_cosine', 'n/a')} "
                f"| {row['negative_refusal_rate']} | {row['negative_refused']} "
                f"| {row['answerable_false_refusal_rate']} | {row['answerable_wrongly_refused']} |\n"
            )

    out.append(
        "\n## 4. 检索层指标与内存库口径的对齐\n\n"
        f"- k={report['k']}，chunk_size={report['chunk_size']}，块数 {report['chunks']}，"
        f"建库 {report['build_seconds']}s\n"
        f"- 可回答题 Hit@K：{report['hit_at_k']}；答案在上下文率：{report['answer_in_context']}\n"
        f"- 与上一轮对齐：`{json.dumps(report['cross_check'], ensure_ascii=False)}`\n"
    )
    return "".join(out)


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="线上 Chroma 口径的阈值拒答实验")
    parser.add_argument("--chunk-size", type=int, default=0, help="0 表示一段一块")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.8,1.0,1.2")
    parser.add_argument("--calibrate", type=int, default=50, help="实测配对的题数")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 道题，冒烟用")
    parser.add_argument(
        "--previous",
        default=str(EVALS_DIR / "results" / "2026-09-29-cmrc.json"),
        help="上一轮内存库结果，用来逐题对齐口径",
    )
    parser.add_argument("--keep", action="store_true", help="保留临时索引目录")
    args = parser.parse_args()

    corpus = read_jsonl(DATASET_DIR / "corpus.jsonl")
    golden = read_jsonl(DATASET_DIR / "golden.jsonl")
    if args.limit:
        golden = golden[: args.limit]
    negatives_total = sum(1 for item in golden if not item["answerable"])
    print(f"语料 {len(corpus)} 段，题目 {len(golden)} 道（含负样本 {negatives_total} 道）\n")

    persist_dir = tempfile.mkdtemp(prefix="cmrc-chroma-threshold-")
    try:
        chunks = to_documents(corpus, args.chunk_size)
        started = time.perf_counter()
        store = build_store(chunks, persist_dir)
        build_seconds = time.perf_counter() - started
        print(f"临时 Chroma 建库 {len(chunks)} 块，{build_seconds:.1f}s，目录 {persist_dir}")

        evidence = read_space_evidence(store)
        print(f"collection 配置：{json.dumps(evidence, ensure_ascii=False)}")

        embeddings = create_embeddings()
        print(f"\n距离口径实测：取 {min(args.calibrate, len(golden))} 道题")
        calibration = calibrate_distance_space(store, embeddings, golden, args.calibrate)
        verdict = calibration["verdict"]
        print(
            f"    口径判定：{verdict['best']}（confirmed={verdict['confirmed']}），"
            f"top1 同段率 {calibration['same_top_doc_count']}，"
            f"语料模长 {calibration['corpus_norm']['min']}~{calibration['corpus_norm']['max']}"
        )

        print(f"\n全量检索 k={args.k} ...")
        started = time.perf_counter()
        rows = probe(store, golden, args.k)
        print(f"    {len(golden)} 道题，{time.perf_counter() - started:.1f}s")
    finally:
        if args.keep:
            print(f"（--keep）临时索引留在 {persist_dir}")
        else:
            shutil.rmtree(persist_dir, ignore_errors=True)

    answerable = [row for row in rows if row["answerable"]]
    negatives = [row for row in rows if not row["answerable"]]
    answerable_top1 = [row["top1_distance"] for row in answerable]
    negative_top1 = [row["top1_distance"] for row in negatives]

    sweep = (
        distance_refusal_sweep(answerable_top1, negative_top1, parse_float_list(args.thresholds))
        if negatives
        else []
    )
    if verdict["confirmed"]:
        for row in sweep:
            row["equivalent_cosine"] = round(
                cosine_from_distance(row["max_distance"], verdict["best"]), 4
            )

    report = {
        "chunk_size": args.chunk_size,
        "k": args.k,
        "chunks": len(chunks),
        "build_seconds": round(build_seconds, 2),
        "n_answerable": len(answerable),
        "n_negative": len(negatives),
        "hit_at_k": round(sum(row["hit"] for row in answerable) / len(answerable), 3)
        if answerable
        else None,
        "answer_in_context": round(
            sum(row["answer_in_context"] for row in answerable) / len(answerable), 3
        )
        if answerable
        else None,
        "space_evidence": evidence,
        "calibration": calibration,
        "answerable_top1_distance": summarize(answerable_top1),
        "negative_top1_distance": summarize(negative_top1),
        "distance_threshold_sweep": sweep,
        "cross_check": cross_check_with_memory(
            rows, Path(args.previous), verdict["best"], args.chunk_size, args.k
        ),
        "per_question": rows,
    }

    stamp = date.today().isoformat()
    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / f"{stamp}-chroma-threshold.json"
    md_path = results_dir / f"{stamp}-chroma-threshold.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(f"\n结果已写入 evals/results/{stamp}-chroma-threshold.{{json,md}}\n")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
