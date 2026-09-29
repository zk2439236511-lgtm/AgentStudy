"""
Document loading and chunking module.

RAG Flow Step 1-2: Load → Chunk
- PyPDFLoader: Extracts text from PDF (one Document per page)
- TextLoader: Reads .txt / .md as one Document (no page numbers)
- RecursiveCharacterTextSplitter: Splits documents into smaller chunks
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# 英文按空格切；splitter 会按顺序尝试这些分隔符，全部失败才退化成按字符硬切
PDF_SEPARATORS = ["\n\n", "\n", " ", ""]

# 中文正文几乎没有空格，沿用上面的分隔符会把句子从中间切断。
# (?<=。) 这类零宽断言在标点**之后**下刀，句号留在前半块，边界干净。
CJK_SEPARATORS = ["\n\n", "\n", "(?<=。)", "(?<=！)", "(?<=？)", "(?<=；)", "(?<=，)", ""]


def _split(
    documents: list,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    is_separator_regex: bool = False,
) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=is_separator_regex,
    )
    return splitter.split_documents(documents)


def load_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """Load a PDF and split it into chunks suitable for embedding.

    PyPDFLoader 每页返回一个 Document，metadata["page"] 是 **0-based** 页号。
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    documents = PyPDFLoader(str(pdf_path)).load()
    print(f"Loaded {len(documents)} pages from {filename}")

    chunks = _split(documents, chunk_size, chunk_overlap, PDF_SEPARATORS)
    for chunk in chunks:
        chunk.metadata["source"] = filename

    print(f"Split {filename} into {len(chunks)} chunks")
    return chunks


def load_and_chunk_text(
    text_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """.txt / .md：整篇只有一个 Document，没有"页"的概念，page 置为 None。"""
    text_path = Path(text_path)
    filename = text_path.name

    documents = TextLoader(str(text_path), encoding="utf-8").load()
    chunks = _split(
        documents, chunk_size, chunk_overlap, CJK_SEPARATORS, is_separator_regex=True
    )
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["page"] = None

    print(f"Split {filename} into {len(chunks)} chunks")
    return chunks


def load_and_chunk_document(
    path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """按后缀分发到对应 loader。未知后缀直接抛错，不静默返回空列表。"""
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return load_and_chunk_pdf(path, chunk_size, chunk_overlap)
    if suffix in {".txt", ".md"}:
        return load_and_chunk_text(path, chunk_size, chunk_overlap)

    raise ValueError(f"不支持的文件类型：{suffix or Path(path).name}")


def load_all_documents(
    directory: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """Load every supported file in a directory and return all chunks."""
    directory = Path(directory)
    files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not files:
        print(f"No supported files found in {directory}")
        return []

    print(f"Found {len(files)} files to load")

    all_chunks = []
    for path in files:
        all_chunks.extend(load_and_chunk_document(path, chunk_size, chunk_overlap))

    print(f"Total: {len(all_chunks)} chunks from {len(files)} files")
    return all_chunks
