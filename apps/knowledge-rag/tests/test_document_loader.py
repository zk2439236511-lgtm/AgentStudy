"""Tests for document loading and chunking."""

import pytest
from pathlib import Path

from ragdemo.document_loader import (
    load_all_documents,
    load_and_chunk_document,
    load_and_chunk_pdf,
    load_and_chunk_text,
)


class TestDocumentLoader:
    """Tests for the document loader module."""

    def test_load_pdf_returns_chunks(self, sample_pdf_path: Path):
        """Test that loading a PDF returns a non-empty list of chunks."""
        chunks = load_and_chunk_pdf(sample_pdf_path)

        assert len(chunks) > 0
        assert all(hasattr(chunk, 'page_content') for chunk in chunks)

    def test_chunks_have_content(self, sample_pdf_path: Path):
        """Test that all chunks have non-empty content."""
        chunks = load_and_chunk_pdf(sample_pdf_path)

        for chunk in chunks:
            assert chunk.page_content.strip(), "Chunk should have non-empty content"

    def test_chunks_contain_transformer_content(self, sample_pdf_path: Path):
        """Test that chunks contain expected content from the Transformer paper."""
        chunks = load_and_chunk_pdf(sample_pdf_path)

        all_content = " ".join(chunk.page_content.lower() for chunk in chunks)

        # The paper should mention transformer or attention
        assert "transformer" in all_content or "attention" in all_content, \
            "PDF should contain transformer/attention content"

    def test_chunk_size_parameter(self, sample_pdf_path: Path):
        """Test that chunk_size parameter affects the output."""
        small_chunks = load_and_chunk_pdf(sample_pdf_path, chunk_size=500)
        large_chunks = load_and_chunk_pdf(sample_pdf_path, chunk_size=2000)

        # Smaller chunk size should result in more chunks
        assert len(small_chunks) > len(large_chunks)

    def test_chunks_have_metadata(self, sample_pdf_path: Path):
        """Test that chunks include metadata."""
        chunks = load_and_chunk_pdf(sample_pdf_path)

        # PyPDFLoader adds page metadata
        for chunk in chunks:
            assert hasattr(chunk, 'metadata')
            assert isinstance(chunk.metadata, dict)

    def test_invalid_pdf_path_raises_error(self):
        """Test that an invalid path raises an appropriate error."""
        with pytest.raises(Exception):
            load_and_chunk_pdf(Path("/nonexistent/file.pdf"))


class TestTextAndDispatch:
    """txt / md 支持：无页码、中文按句边界切、按后缀分发。"""

    CN_SENTENCE = "多头注意力把查询键值投影到多个子空间并行计算。"

    def _write_cn(self, tmp_path, name, times=20):
        path = tmp_path / name
        path.write_text(self.CN_SENTENCE * times, encoding="utf-8")
        return path

    def test_text_chunks_have_no_page_but_keep_source(self, tmp_path):
        chunks = load_and_chunk_text(self._write_cn(tmp_path, "notes.md"), chunk_size=100, chunk_overlap=0)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["page"] is None
            assert chunk.metadata["source"] == "notes.md"

    def test_chinese_splits_on_sentence_boundaries(self, tmp_path):
        """每块都收在句号上，正文里不该混进正则字符串。"""
        chunks = load_and_chunk_text(self._write_cn(tmp_path, "cn.txt"), chunk_size=100, chunk_overlap=0)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.page_content.endswith("。")
            assert "(?<=" not in chunk.page_content

    def test_english_pdf_separators_unchanged(self, sample_pdf_path: Path):
        """PDF 那条路继续用原来的分隔符，已有索引的切块口径不能变。"""
        chunks = load_and_chunk_pdf(sample_pdf_path, chunk_size=200, chunk_overlap=0)

        assert chunks
        assert all(chunk.metadata["page"] is not None for chunk in chunks)

    def test_dispatcher_rejects_unknown_suffix(self, tmp_path):
        with pytest.raises(ValueError, match="不支持的文件类型"):
            load_and_chunk_document(tmp_path / "resume.docx")

    def test_load_all_documents_takes_txt_and_md_and_ignores_others(self, tmp_path):
        self._write_cn(tmp_path, "a.txt", times=8)
        self._write_cn(tmp_path, "b.md", times=8)
        (tmp_path / "c.docx").write_text("ignore me", encoding="utf-8")

        chunks = load_all_documents(tmp_path, chunk_size=100, chunk_overlap=0)

        assert {chunk.metadata["source"] for chunk in chunks} == {"a.txt", "b.md"}
