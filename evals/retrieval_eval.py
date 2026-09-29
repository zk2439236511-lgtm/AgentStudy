"""检索评测：chunk_size × k 网格实验。

用法：
    python evals/retrieval_eval.py
    python evals/retrieval_eval.py --chunk-sizes 500,1000 --ks 4

每组建一个独立的内存向量库（不碰 data/chroma 里的持久化索引），
把 evals/questions.json 全跑一遍，结果写进 evals/results/。
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
APP_DIR = REPO_ROOT / "apps" / "knowledge-rag"
PDF_PATH = APP_DIR / "documents" / "sample.pdf"

sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(APP_DIR / "src"))

from dotenv import load_dotenv

load_dotenv(APP_DIR / ".env")

from metrics import evaluate
from ragdemo.document_loader import load_and_chunk_pdf
from ragdemo.vector_store import create_vector_store


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def run_config(questions: list[dict], chunk_size: int, k: int) -> dict:
    chunks = load_and_chunk_pdf(PDF_PATH, chunk_size=chunk_size)

    started = time.perf_counter()
    store = create_vector_store(chunks)
    build_seconds = time.perf_counter() - started

    cases = []
    for item in questions:
        docs = store.similarity_search(item["question"], k=k)
        cases.append(
            {
                "retrieved": [doc.metadata.get("page") for doc in docs],
                "expected": item["expected_pages"],
                "keywords_retrieved": [doc.page_content for doc in docs],
                "expected_keywords": item["expected_keywords"],
            }
        )

    result = evaluate(cases)
    result.update(
        {
            "chunk_size": chunk_size,
            "k": k,
            "chunks": len(chunks),
            "build_seconds": round(build_seconds, 2),
        }
    )
    return result


def to_markdown(rows: list[dict]) -> str:
    header = (
        "| chunk_size | k | 块数 | 建库耗时(s) | Hit@K | Recall@K | Precision@K | nDCG@K | MRR | 关键词覆盖 |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['chunk_size']} | {r['k']} | {r['chunks']} | {r['build_seconds']} "
        f"| {r['hit_rate']} | {r['recall']} | {r['precision']} | {r['ndcg']} "
        f"| {r['mrr']} | {r['keyword_coverage']} |\n"
        for r in rows
    )
    return header + body


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--chunk-sizes", default="500,1000,2000")
    parser.add_argument("--ks", default="4,8")
    args = parser.parse_args()

    questions = json.loads((EVALS_DIR / "questions.json").read_text(encoding="utf-8"))
    print(f"评测集：{len(questions)} 个问题\n")

    rows = []
    for chunk_size in parse_int_list(args.chunk_sizes):
        for k in parse_int_list(args.ks):
            print(f"--- chunk_size={chunk_size}, k={k} ---")
            row = run_config(questions, chunk_size, k)
            rows.append(row)
            print(
                f"    块数 {row['chunks']}，建库 {row['build_seconds']}s，"
                f"Hit@K {row['hit_rate']}，Recall@K {row['recall']}，"
                f"MRR {row['mrr']}，关键词覆盖 {row['keyword_coverage']}\n"
            )

    stamp = date.today().isoformat()
    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"{stamp}-grid.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / f"{stamp}-grid.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"结果已写入 evals/results/{stamp}-grid.{{json,md}}\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
