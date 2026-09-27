"""跑通 RAG 基线：Load → Chunk → Embed → Retrieve → Generate（百炼 Qwen）。

用法：
    python examples/rag_baseline_demo.py
    python examples/rag_baseline_demo.py "What is multi-head attention?"
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "apps" / "knowledge-rag"
sys.path.insert(0, str(APP_DIR / "src"))

from dotenv import load_dotenv

load_dotenv(APP_DIR / ".env")

from ragdemo.document_loader import load_and_chunk_pdf
from ragdemo.rag_chain import create_rag_chain_with_sources, print_source_documents
from ragdemo.vector_store import create_vector_store

question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "How does self-attention work in the Transformer?"
)

chunks = load_and_chunk_pdf(APP_DIR / "documents" / "sample.pdf")
vector_store = create_vector_store(chunks)
chain = create_rag_chain_with_sources(vector_store)

print("\nQ:", question, "\n")
result = chain.invoke(question)
print("A:", result["answer"])
print_source_documents(result["source_documents"])
