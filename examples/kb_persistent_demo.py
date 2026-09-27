"""阶段 3 验证：索引建一次、重启还能问。

用法：
    python examples/kb_persistent_demo.py
    python examples/kb_persistent_demo.py "What is multi-head attention?"

第一次运行：解析 PDF + 向量化入库（花钱、花时间）。
第二次运行：登记表发现内容没变 → 跳过 embedding，直接从磁盘索引问答。
"""

import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "apps" / "knowledge-rag"
sys.path.insert(0, str(APP_DIR / "src"))

from dotenv import load_dotenv

load_dotenv(APP_DIR / ".env")

from ragdemo.ingest import (
    count_vectors,
    ingest_directory,
    init_registry,
    load_vector_store,
)
from ragdemo.rag_chain import create_rag_chain_with_sources, print_source_documents

PERSIST_DIR = APP_DIR / "data" / "chroma"
DB_PATH = APP_DIR / "data" / "registry.db"

conn = init_registry(DB_PATH)

started = time.perf_counter()
result = ingest_directory(APP_DIR / "documents", conn, PERSIST_DIR)
ingest_seconds = time.perf_counter() - started

store = load_vector_store(PERSIST_DIR)
print(
    f"\n入库耗时 {ingest_seconds:.2f}s：新处理 {len(result['ingested'])} 个文件、"
    f"跳过 {len(result['skipped'])} 个未变文件、新增 {result['chunks_added']} 块；"
    f"磁盘索引现有 {count_vectors(store)} 个向量"
)

question = (
    sys.argv[1] if len(sys.argv) > 1 else "What is multi-head attention?"
)
chain = create_rag_chain_with_sources(store)

started = time.perf_counter()
answer = chain.invoke(question)
print(f"\nQ: {question}\n")
print(f"A: {answer['answer']}\n")
print(f"（回答耗时 {time.perf_counter() - started:.2f}s）")
print_source_documents(answer["source_documents"])

conn.close()
