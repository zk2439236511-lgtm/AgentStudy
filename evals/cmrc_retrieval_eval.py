"""CMRC2018 外部金标集上的检索评测。

和 evals/retrieval_eval.py 的区别只有一个，但是关键的那个：
标准答案来自第三方人工标注（见 evals/datasets/cmrc2018/MANIFEST.json），
不是"先跑一遍检索看命中哪些页"自己填出来的。

用法：
    python evals/cmrc_retrieval_eval.py --limit 20          # 先小规模冒烟
    python evals/cmrc_retrieval_eval.py                     # 全量
    python evals/cmrc_retrieval_eval.py --chunk-sizes 0,500 --ks 4,8

会真实调用百炼 embedding（每 10 块一次请求），结果写进 evals/results/。

注意口径：这里用的是 InMemoryVectorStore，分数是**余弦相似度**（越大越相关）；
线上 kb-web 用的是 Chroma，透出的是**向量距离**（越小越相关）。两者别混着说。
"""

import argparse
import json
import statistics
import sys
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
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(APP_DIR / ".env")

from metrics import hit_rate, ndcg, precision, recall, similarity_refusal_sweep
from ragdemo.document_loader import CJK_SEPARATORS
from ragdemo.vector_store import EMBEDDING_BATCH_SIZE, create_embeddings


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path}，先运行 python evals/datasets/build_cmrc_golden.py"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def to_documents(corpus: list[dict], chunk_size: int) -> list[Document]:
    """chunk_size=0 表示一段一块；给了正值就按中文句末分隔符再切。"""
    documents = [
        Document(
            page_content=doc["text"],
            metadata={"doc_id": doc["doc_id"], "title": doc["title"]},
        )
        for doc in corpus
    ]
    if not chunk_size:
        return documents

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        separators=CJK_SEPARATORS,
        is_separator_regex=True,
    )
    chunks = []
    for document in documents:
        for part in splitter.split_text(document.page_content):
            chunks.append(Document(page_content=part, metadata=dict(document.metadata)))
    return chunks


def build_store(chunks: list[Document]) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=create_embeddings())
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        store.add_documents(chunks[start : start + EMBEDDING_BATCH_SIZE])
    return store


def retrieve(store: InMemoryVectorStore, question: str, k: int) -> tuple[list[str], list[str], float]:
    """返回 (按序 doc_id 列表, 按序正文列表, top1 相似度)。"""
    hits = store.similarity_search_with_score(question, k=k)
    doc_ids = [doc.metadata["doc_id"] for doc, _ in hits]
    texts = [doc.page_content for doc, _ in hits]
    top1 = float(hits[0][1]) if hits else 0.0
    return doc_ids, texts, top1


def summarize(values: list[float]) -> dict:
    """分位数摘要。下标用「四舍五入到最近的那个样本」而不是向下取整：

    向下取整在 n=2 时会让 p75 == min，渲染出来就是「中位 7.4s / p75 3.3s」这种反序数字，
    看着像统计炸了。改成最近下标后 min ≤ p25 ≤ median ≤ p75 ≤ max 恒成立。
    """
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    last = len(ordered) - 1

    def pick(quantile: float) -> float:
        return ordered[min(int(round(last * quantile)), last)]

    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p25": round(pick(0.25), 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(pick(0.75), 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


def threshold_sweep(
    answerable_top1: list[float], negatives_top1: list[float], thresholds: list[float]
) -> list[dict]:
    """转发给 metrics.similarity_refusal_sweep；保留这个名字是为了不改动结果文件的字段语义。

    只是给 ④ 挑候选阈值用的分布证据，不是调参结论：这里的分数是内存库的余弦
    相似度，线上是 Chroma 距离，阈值搬不过去。
    """
    return similarity_refusal_sweep(answerable_top1, negatives_top1, thresholds)


def score_config(
    store: InMemoryVectorStore,
    golden: list[dict],
    chunk_size: int,
    n_chunks: int,
    build_seconds: float,
    k: int,
    thresholds: list[float],
) -> dict:
    """在同一个已建好的向量库上换一个 k 打分。

    建库要花钱，所以外层按 chunk_size 建一次、内层对每个 k 只跑检索。
    """
    answerables = [row for row in golden if row["answerable"]]
    negatives = [row for row in golden if not row["answerable"]]

    metrics_cases, per_question = [], []
    answerable_top1, negative_top1 = [], []

    for row in answerables:
        doc_ids, texts, top1 = retrieve(store, row["question"], k)
        gold = set(row["gold_doc_ids"])
        rank = next((i + 1 for i, doc in enumerate(doc_ids) if doc in gold), None)
        answerable_top1.append(top1)
        metrics_cases.append(
            {
                "retrieved": doc_ids,
                "expected": row["gold_doc_ids"],
                "texts": texts,
                "answers": row["gold_answers"],
                "rank": rank,
            }
        )
        per_question.append(
            {
                "question_id": row["question_id"],
                "answerable": True,
                "top1_similarity": round(top1, 4),
                "gold_rank": rank,
            }
        )

    for row in negatives:
        doc_ids, _texts, top1 = retrieve(store, row["question"], k)
        negative_top1.append(top1)
        per_question.append(
            {
                "question_id": row["question_id"],
                "answerable": False,
                "top1_similarity": round(top1, 4),
                "top1_doc_id": doc_ids[0] if doc_ids else None,
            }
        )

    n = len(metrics_cases)
    hits = [hit_rate(c["retrieved"], c["expected"]) for c in metrics_cases]
    recalls = [recall(c["retrieved"], c["expected"]) for c in metrics_cases]
    precisions = [precision(c["retrieved"], c["expected"]) for c in metrics_cases]
    ndcgs = [ndcg(c["retrieved"], c["expected"]) for c in metrics_cases]
    mrrs = [1 / c["rank"] if c["rank"] else 0.0 for c in metrics_cases]
    # 答案字面有没有出现在召回文本里：这是"上下文够不够答题"的下界，
    # 不等于答案正确率，模型可能用同义表述答对
    in_context = [
        1.0 if any(answer in "\n".join(c["texts"]) for answer in c["answers"]) else 0.0
        for c in metrics_cases
    ]

    return {
        "chunk_size": chunk_size,
        "k": k,
        "chunks": n_chunks,
        "build_seconds": round(build_seconds, 2),
        "n_answerable": n,
        "n_negative": len(negatives),
        "hit_at_k": round(sum(hits) / n, 3),
        "recall_at_k": round(sum(recalls) / n, 3),
        "precision_at_k": round(sum(precisions) / n, 3),
        "ndcg_at_k": round(sum(ndcgs) / n, 3),
        "mrr": round(sum(mrrs) / n, 3),
        "answer_in_context": round(sum(in_context) / n, 3),
        "answerable_top1_similarity": summarize(answerable_top1),
        "negative_top1_similarity": summarize(negative_top1),
        "threshold_sweep": threshold_sweep(answerable_top1, negative_top1, thresholds)
        if negatives
        else [],
        "per_question": per_question,
    }


def to_markdown(rows: list[dict]) -> str:
    out = [
        "## 检索层指标（标准答案来自 CMRC2018 人工标注，非自造）\n\n",
        "| chunk_size | k | 块数 | 建库(s) | Hit@K | Recall@K | Precision@K | nDCG@K | MRR | 答案在上下文 |\n",
        "|---|---|---|---|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        out.append(
            f"| {r['chunk_size']} | {r['k']} | {r['chunks']} | {r['build_seconds']} "
            f"| {r['hit_at_k']} | {r['recall_at_k']} | {r['precision_at_k']} | {r['ndcg_at_k']} "
            f"| {r['mrr']} | {r['answer_in_context']} |\n"
        )

    out.append(
        "\n## top1 余弦相似度分布（越大越相关；内存库口径，不是线上 Chroma 的距离）\n\n"
        "| chunk_size | k | 可回答 n | min | p25 | 中位 | p75 | max | 负样本 n | min | p25 | 中位 | p75 | max |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in rows:
        pos, neg = r["answerable_top1_similarity"], r["negative_top1_similarity"]
        out.append(
            f"| {r['chunk_size']} | {r['k']} "
            f"| {pos['n']} | {pos['min']} | {pos['p25']} | {pos['median']} | {pos['p75']} | {pos['max']} "
            f"| {neg['n']} | {neg['min']} | {neg['p25']} | {neg['median']} | {neg['p75']} | {neg['max']} |\n"
        )

    if rows and rows[0]["threshold_sweep"]:
        out.append(
            "\n## 阈值扫描：把「top1 相似度低于 t 就拒答」当成分类器\n\n"
            "| t | 负样本正确拒答率 | 负样本 | 可回答题误拒率 | 可回答题 |\n"
            "|---|---|---|---|---|\n"
        )
        for sweep in rows[0]["threshold_sweep"]:
            out.append(
                f"| {sweep['threshold']} | {sweep['negative_refusal_rate']} "
                f"| {sweep['negative_refused']} | {sweep['answerable_false_refusal_rate']} "
                f"| {sweep['answerable_wrongly_refused']} |\n"
            )
        out.append(
            "\n（只统计了第一组配置；这一层是给 ④ 挑候选阈值，不构成"
            "「阈值拒答有效」的结论——线上是 Chroma 距离，口径不同）\n"
        )
    return "".join(out)


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CMRC2018 金标集检索评测")
    parser.add_argument("--chunk-sizes", default="0,500", help="0 表示一段一块")
    parser.add_argument("--ks", default="4,8")
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 道题，冒烟用")
    args = parser.parse_args()

    corpus = read_jsonl(DATASET_DIR / "corpus.jsonl")
    golden = read_jsonl(DATASET_DIR / "golden.jsonl")
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    if args.limit:
        golden = golden[: args.limit]

    print(
        f"语料 {len(corpus)} 段，题目 {len(golden)} 道（含负样本 "
        f"{sum(1 for g in golden if not g['answerable'])} 道）\n"
    )

    rows = []
    for chunk_size in parse_int_list(args.chunk_sizes):
        chunks = to_documents(corpus, chunk_size)
        print(f"--- 建库 chunk_size={chunk_size}（{len(chunks)} 块）---")
        started = time.perf_counter()
        store = build_store(chunks)
        build_seconds = time.perf_counter() - started
        print(f"    建库 {build_seconds:.1f}s，同一份索引上跑所有 k\n")

        for k in parse_int_list(args.ks):
            row = score_config(store, golden, chunk_size, len(chunks), build_seconds, k, thresholds)
            rows.append(row)
            print(
                f"    k={k}：Hit {row['hit_at_k']}，Recall {row['recall_at_k']}，"
                f"Precision {row['precision_at_k']}，nDCG {row['ndcg_at_k']}，"
                f"MRR {row['mrr']}，答案在上下文 {row['answer_in_context']}"
            )
        print()

    stamp = date.today().isoformat()
    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"{stamp}-cmrc.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / f"{stamp}-cmrc.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"结果已写入 evals/results/{stamp}-cmrc.{{json,md}}\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
