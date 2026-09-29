"""金标集抽取器的测试。

分两层：一层用合成数据验证 build() 的规则（确定性、span 校验、负样本真的查不到），
一层直接检查**已经提交进仓库**的 corpus.jsonl / golden.jsonl 自不自洽——
抽取逻辑改了但忘了重跑产物时，这一层会红。
"""

import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(EVALS_DIR / "datasets"))

from build_cmrc_golden import build, is_usable, load_paragraphs, sha256_of  # noqa: E402

DATASET_DIR = EVALS_DIR / "datasets" / "cmrc2018"


def make_paragraph(index: int, n_questions: int = 2) -> dict:
    """每段答案都带自己的编号，保证不同段落之间不会字面重复。"""
    context_id = f"C_{index:03d}"
    return {
        "context_id": context_id,
        "title": f"标题{index}",
        "context_text": f"这是第{index}段原文，里面写着答案{index}。",
        "qas": [
            {
                "query_id": f"{context_id}_Q{j}",
                "query_text": f"第{index}段的问题{j}？",
                "answers": [f"答案{index}"],
            }
            for j in range(n_questions)
        ],
    }


def synthetic_paras(count: int) -> list[dict]:
    return [make_paragraph(i) for i in range(count)]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_build(paras=None, **overrides):
    params = {"seed": 7, "n_contexts": 10, "q_per_context": 2, "n_unanswerable": 3}
    params.update(overrides)
    return build(paras if paras is not None else synthetic_paras(40), **params)


def test_same_seed_same_output():
    """抽样必须由 seed 完全决定，否则 MANIFEST 里的复现声明是空的。"""
    first_corpus, first_golden, _ = run_build()
    second_corpus, second_golden, _ = run_build()

    assert first_corpus == second_corpus
    assert first_golden == second_golden


def test_different_seed_draws_a_different_sample():
    a, _, _ = run_build(seed=7)
    b, _, _ = run_build(seed=2026)

    assert [doc["doc_id"] for doc in a] != [doc["doc_id"] for doc in b]


def test_answerable_rows_point_at_a_real_span_of_their_own_doc():
    corpus, golden, _ = run_build()
    by_doc = {doc["doc_id"]: doc["text"] for doc in corpus}
    answerable = [row for row in golden if row["answerable"]]

    assert len(answerable) == 20
    for row in answerable:
        assert len(row["gold_doc_ids"]) == 1
        text = by_doc[row["gold_doc_ids"][0]]
        assert row["gold_answers"], row
        assert all(answer in text for answer in row["gold_answers"])


def test_constructed_negatives_really_cannot_be_answered_from_the_corpus():
    corpus, golden, _ = run_build()
    doc_ids = {doc["doc_id"] for doc in corpus}
    texts = [doc["text"] for doc in corpus]
    negatives = [row for row in golden if not row["answerable"]]

    assert len(negatives) == 3
    for row in negatives:
        assert row["gold_doc_ids"] == []
        # 出题用的段落被排除在语料之外，且这个排除关系可追溯
        assert row["withheld_doc_id"] not in doc_ids
        assert not any(answer in text for text in texts for answer in row["gold_answers"])


def test_negatives_and_answerables_are_shuffled_together():
    """两类题不能按"可回答在前、负样本在后"分组排列——分组会让人工看一眼就产生偏向。"""
    _, golden, _ = run_build()
    flags = [row["answerable"] for row in golden]

    assert flags != sorted(flags, reverse=True), "负样本全堆在末尾，说明没打散"
    assert flags != sorted(flags), "负样本全堆在开头，说明没打散"


def test_is_usable_rejects_excel_mangled_answers():
    """CMRC2018 里有 27 道题的答案被 Excel 改成日期序列号，这种段落不能进抽样池。"""
    paragraph = make_paragraph(1)
    paragraph["qas"][0]["answers"] = ["答案1", 39764.0]

    assert not is_usable(paragraph, 2)


def test_is_usable_rejects_answer_not_present_in_text():
    paragraph = make_paragraph(1)
    paragraph["qas"][0]["answers"] = ["原文里根本没有的说法"]

    assert not is_usable(paragraph, 2)


def test_is_usable_rejects_too_few_questions():
    assert not is_usable(make_paragraph(1, n_questions=1), 2)


def test_load_paragraphs_rejects_checksum_mismatch(tmp_path):
    raw = tmp_path / "cmrc.json"
    raw.write_text(json.dumps(synthetic_paras(3)), encoding="utf-8")

    try:
        load_paragraphs(raw, expected_sha256="0" * 64)
    except ValueError as cause:
        assert "校验和不匹配" in str(cause)
    else:
        raise AssertionError("校验和不匹配时应该报错，而不是继续往下抽")


def test_shipped_corpus_and_golden_are_self_consistent():
    """直接检查提交进仓库的产物：抽取规则改了但忘了重跑，这里会红。"""
    corpus = read_jsonl(DATASET_DIR / "corpus.jsonl")
    golden = read_jsonl(DATASET_DIR / "golden.jsonl")
    by_doc = {doc["doc_id"]: doc["text"] for doc in corpus}
    texts = list(by_doc.values())

    assert len(corpus) == 100, "语料段数要和 MANIFEST 声明的一致"
    assert len(by_doc) == len(corpus), "doc_id 重复会让 gold_doc_ids 失去意义"
    assert len(golden) == 230

    answerables = [row for row in golden if row["answerable"]]
    negatives = [row for row in golden if not row["answerable"]]
    assert (len(answerables), len(negatives)) == (200, 30)

    for row in answerables:
        text = by_doc[row["gold_doc_ids"][0]]
        assert all(answer in text for answer in row["gold_answers"])
    for row in negatives:
        assert row["gold_doc_ids"] == []
        assert not any(a in text for text in texts for a in row["gold_answers"])


def test_shipped_manifest_matches_the_artifacts():
    manifest = json.loads((DATASET_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    corpus = read_jsonl(DATASET_DIR / "corpus.jsonl")
    golden = read_jsonl(DATASET_DIR / "golden.jsonl")

    assert manifest["counts"] == {"corpus": len(corpus), "golden": len(golden)}
    assert manifest["params"]["n_contexts"] == len(corpus)
    assert manifest["params"]["n_unanswerable"] == sum(1 for r in golden if not r["answerable"])
    # 产物内容变了却没更新校验和，说明有人手改了金标集
    assert manifest["artifacts"]["corpus.jsonl"] == sha256_of(DATASET_DIR / "corpus.jsonl")
    assert manifest["artifacts"]["golden.jsonl"] == sha256_of(DATASET_DIR / "golden.jsonl")
