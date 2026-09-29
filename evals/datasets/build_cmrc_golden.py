"""从 CMRC2018 原始数据里抽一份**外部**检索金标集。

为什么要有这个脚本：evals/questions.json 的 expected_pages 是"先跑检索看命中哪些页、
再人工核对"得来的，等于部分标准答案出自被评测的那个检索器本身（数据泄漏）。
CMRC2018 是第三方标注的 SQuAD 式中文数据集，问题—段落—答案由人工标好，
拿它当 ground truth 才谈得上"外部"。

用法（先把原始文件放进 evals/datasets/raw/，见 MANIFEST 里的 URL + sha256）：
    python evals/datasets/build_cmrc_golden.py
    python evals/datasets/build_cmrc_golden.py --n-contexts 30 --n-unanswerable 10

产出（都在 evals/datasets/cmrc2018/ 下）：
    corpus.jsonl    100 段中文语料，评测时喂给向量库
    golden.jsonl    200 道可回答 + 30 道构造式不可回答，字段含 gold_doc_ids
    MANIFEST.json   数据来源、参数、条数、校验和——复现要用

全程不联网、不调模型，抽样由 seed 固定，两次运行结果逐字节相同。
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent
RAW_DIR = DATASETS_DIR / "raw"
OUT_DIR = DATASETS_DIR / "cmrc2018"

SOURCE = {
    "name": "CMRC2018 (cmrc2018_dev)",
    "repo": "https://github.com/ymcui/cmrc2018",
    "url": "https://raw.githubusercontent.com/ymcui/cmrc2018/master/data/cmrc2018_dev.json",
    "license": "CC BY-SA 4.0",
    "sha256": "5cfe4414c28a8ecbb51670f78c0dc7d1049f286c2d5769b52f1f94bcc0752cf1",
}

# CMRC2018 公开数据只有可回答题（三个 split 都没有 is_impossible 字段），
# 所以负样本只能自己构造，provenance 必须写清楚，不能冒充数据集提供
PROVENANCE_ANSWERABLE = "cmrc2018_dev_human_annotated"
PROVENANCE_UNANSWERABLE = "constructed_absent_context"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_paragraphs(raw_path: Path, expected_sha256: str | None = None) -> list[dict]:
    """读原始段落列表；校验和不匹配就拒绝，免得拿错版本还自称可复现。"""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"缺少原始语料 {raw_path}\n"
            f"请从 {SOURCE['url']} 下载后放到 {RAW_DIR}/（需走本机代理）"
        )

    digest = sha256_of(raw_path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"校验和不匹配：期望 {expected_sha256[:12]}，实际 {digest[:12]}")

    return json.loads(raw_path.read_text(encoding="utf-8"))


def is_usable(paragraph: dict, min_questions: int) -> bool:
    """段落可用的判据：问数够，且每个答案都是原文里真实存在的字面 span。

    答案不在原文里的标注没法用检索层指标判定，只能整段丢掉。
    CMRC2018 公开数据里有 27 道题的某个答案被 Excel 改坏了（日期成 39764.0 这种
    序列号、"147位" 成 147.0），这类段落一并排除，并把这个数量记进 MANIFEST。
    """
    qas = paragraph.get("qas") or []
    if len(qas) < min_questions:
        return False
    text = paragraph["context_text"]
    return all(
        isinstance(answer, str) and answer.strip() and answer in text
        for qa in qas
        for answer in qa.get("answers") or []
    )


def count_dirty_answers(paragraphs: list[dict]) -> int:
    """有多少道题带非字符串答案（Excel 化脏标注），供 MANIFEST 记录。"""
    return sum(
        1
        for p in paragraphs
        for qa in p.get("qas") or []
        if any(not isinstance(a, str) for a in qa.get("answers") or [])
    )


def leaks_into(answers: list[str], corpus_texts: list[str]) -> bool:
    """构造负样本的体检：它的答案字面出现在语料里，就不算"查不到"。"""
    return any(answer in text for text in corpus_texts for answer in answers)


def build(
    paragraphs: list[dict],
    seed: int,
    n_contexts: int,
    q_per_context: int,
    n_unanswerable: int,
) -> tuple[list[dict], list[dict], dict]:
    eligible = sorted(
        (p for p in paragraphs if is_usable(p, q_per_context)),
        key=lambda p: p["context_id"],
    )
    rng = random.Random(seed)
    if n_contexts > len(eligible):
        raise ValueError(f"可用段落只有 {len(eligible)} 段，不够抽 {n_contexts} 段")

    picked = rng.sample(eligible, n_contexts)

    corpus, dropped_duplicates = [], 0
    for paragraph in picked:
        if any(paragraph["context_text"] == doc["text"] for doc in corpus):
            dropped_duplicates += 1
            continue
        corpus.append(
            {
                "doc_id": paragraph["context_id"],
                "title": paragraph.get("title", ""),
                "text": paragraph["context_text"],
            }
        )

    in_corpus = {doc["text"] for doc in corpus}
    answerable = []
    for paragraph in picked:
        if paragraph["context_text"] not in in_corpus:
            continue
        for qa in paragraph["qas"][:q_per_context]:
            answerable.append(
                {
                    "question_id": qa["query_id"],
                    "question": qa["query_text"],
                    "answerable": True,
                    "gold_doc_ids": [paragraph["context_id"]],
                    "gold_answers": sorted(set(qa["answers"])),
                    "provenance": PROVENANCE_ANSWERABLE,
                }
            )

    # 负样本：问题本身是真人标的，但它所属的段落**没进语料**，
    # 所以对这份语料而言正确答案就是"查不到"
    negatives_pool = sorted(
        (
            p
            for p in eligible
            if p["context_text"] not in in_corpus
        ),
        key=lambda p: p["context_id"],
    )
    rng.shuffle(negatives_pool)

    unanswerable, dropped_leaks = [], 0
    corpus_texts = [doc["text"] for doc in corpus]
    for paragraph in negatives_pool:
        if len(unanswerable) >= n_unanswerable:
            break
        qa = paragraph["qas"][0]
        question = {
            "question_id": qa["query_id"],
            "question": qa["query_text"],
            "answerable": False,
            "gold_doc_ids": [],
            "gold_answers": sorted(set(qa["answers"])),
            "provenance": PROVENANCE_UNANSWERABLE,
            "withheld_doc_id": paragraph["context_id"],
        }
        if leaks_into(question["gold_answers"], corpus_texts):
            dropped_leaks += 1
            continue
        unanswerable.append(question)

    if len(unanswerable) < n_unanswerable:
        raise ValueError(
            f"只凑到 {len(unanswerable)} 道干净负样本，不足 {n_unanswerable}"
            f"（{dropped_leaks} 道因答案字面出现在语料里被丢弃）"
        )

    golden = answerable + unanswerable
    # 打散：两类混在一起，避免任何人按顺序一眼看出哪道是送分题
    rng.shuffle(golden)

    stats = {
        "paragraphs_in_split": len(paragraphs),
        "dirty_answer_items": count_dirty_answers(paragraphs),
        "eligible_paragraphs": len(eligible),
        "corpus_docs": len(corpus),
        "dropped_duplicate_texts": dropped_duplicates,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "dropped_leaking_negatives": dropped_leaks,
    }
    return corpus, golden, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 CMRC2018 检索金标集")
    parser.add_argument("--raw", default=str(RAW_DIR / "cmrc2018_dev.json"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--n-contexts", type=int, default=100)
    parser.add_argument("--q-per-context", type=int, default=2)
    parser.add_argument("--n-unanswerable", type=int, default=30)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    paragraphs = load_paragraphs(raw_path, SOURCE["sha256"])
    corpus, golden, stats = build(
        paragraphs,
        seed=args.seed,
        n_contexts=args.n_contexts,
        q_per_context=args.q_per_context,
        n_unanswerable=args.n_unanswerable,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "corpus.jsonl", corpus)
    write_jsonl(out_dir / "golden.jsonl", golden)

    manifest = {
        "source": SOURCE,
        "raw_file": raw_path.name,
        "raw_sha256": sha256_of(raw_path),
        "params": {
            "seed": args.seed,
            "n_contexts": args.n_contexts,
            "q_per_context": args.q_per_context,
            "n_unanswerable": args.n_unanswerable,
        },
        "stats": stats,
        "counts": {"corpus": len(corpus), "golden": len(golden)},
        "artifacts": {
            "corpus.jsonl": sha256_of(out_dir / "corpus.jsonl"),
            "golden.jsonl": sha256_of(out_dir / "golden.jsonl"),
        },
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveats": [
            "CMRC2018 公开数据不含 unanswerable 题；answerable=false 的 30 道是构造的"
            "（问题为真人标注，但其所属段落被排除在语料外），不是数据集原生负样本",
            "gold_doc_ids 只标了标注来源段落；同一段答案字面出现在其他段落时不会追加 gold，"
            "构造负样本阶段已用答案字面泄漏检查过滤，可回答题未做该检查",
            f"有 {stats['dirty_answer_items']} 道题的某个答案是 Excel 化的非字符串值"
            "（日期成序列号、数字丢单位），其所在段落整体排除在抽样池之外",
        ],
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n已写入 {out_dir}/corpus.jsonl, golden.jsonl, MANIFEST.json")


if __name__ == "__main__":
    main()
