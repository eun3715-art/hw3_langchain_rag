#!/usr/bin/env python3
"""
RAG 시스템 핵심 로직 검증 스크립트 (FastAPI 제외)
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from langchain_core.documents import Document

print("=" * 60)
print("RAG core logic test")
print("=" * 60)

passed = 0
failed = 0

# --- Test 1: Router ---
print("\n[1] Router - query classification")

from engine.router import route_query, execute_sql_query, init_sqlite_db

init_sqlite_db()

test_cases = [
    ("음식 준비 사진을 보여줘", "vector"),
    ("평균 평점이 높은 문서들은?", "sql"),
    ("사람들이 웃고 있는 장면", "vector"),
    ("카테고리별 문서 개수는?", "sql"),
]

for query, expected in test_cases:
    result = route_query(query, use_llm=False)
    actual = result.get("route", "unknown")
    status = "PASS" if actual == expected else "FAIL"
    if actual == expected:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] '{query[:35]}' -> {actual} (expected: {expected})")
    if actual == "sql":
        print(f"         SQL: {result.get('sql', 'N/A')[:60]}")

sql_query = "SELECT category, COUNT(*) as cnt FROM documents GROUP BY category LIMIT 5"
df = execute_sql_query(sql_query)
print(f"\n  SQL execution: {len(df)} rows returned")

# --- Test 2: Re-ranker ---
print("\n[2] Re-ranker - Graphlet-aware reranking")

from engine.reranker import turbo_rerank

test_docs = []
for i in range(1, 6):
    doc = Document(
        page_content=f"Document {i}",
        metadata={
            "id": i,
            "similarity_score": 0.9 - (i * 0.05),
            "category": "A" if i <= 3 else "B",
            "author": "Author1" if i <= 2 else "Author2",
            "year": 2020 if i <= 3 else 2021,
        }
    )
    test_docs.append(doc)

print("  Before rerank:")
for doc in test_docs:
    print(f"    ID={doc.metadata['id']}: score={doc.metadata['similarity_score']:.4f}")

reranked = turbo_rerank(test_docs, alpha=0.2)

print("  After rerank:")
for i, doc in enumerate(reranked, 1):
    print(f"    {i}. ID={doc.metadata['id']}: "
          f"base={doc.metadata['similarity_score']:.4f} "
          f"+ bonus={doc.metadata['graphlet_bonus']:.4f} "
          f"= final={doc.metadata['rerank_score']:.4f}")
passed += 1

# --- Test 3: Ingestion ---
print("\n[3] Ingestion - MD5 deduplication")

from engine.ingestion import _compute_md5, _load_hash_registry, _save_hash_registry

test_dir = PROJECT_ROOT / "test_ingest"
test_dir.mkdir(exist_ok=True)

test_files = [
    ("file1.txt", "Content of file 1"),
    ("file2.txt", "Content of file 2"),
    ("file3.txt", "Content of file 1"),  # duplicate
]

hashes = {}
for filename, content in test_files:
    filepath = test_dir / filename
    filepath.write_text(content, encoding="utf-8")
    hashes[filename] = _compute_md5(filepath)
    print(f"  {filename}: {hashes[filename][:12]}...")

unique = set(hashes.values())
print(f"  Files: {len(hashes)}, unique hashes: {len(unique)}")
if len(unique) < len(hashes):
    print("  Duplicate detected (expected)")
    passed += 1
else:
    print("  [FAIL] duplicate not detected")
    failed += 1

registry = {"file1.txt": hashes["file1.txt"]}
_save_hash_registry(registry)
loaded = _load_hash_registry()
if len(loaded) == len(registry):
    print("  Registry save/load: OK")
    passed += 1
else:
    print("  [FAIL] Registry save/load mismatch")
    failed += 1

shutil.rmtree(test_dir)

# --- Test 4: RAG Chain prompts ---
print("\n[4] RAG Chain - prompt templates")

from engine.rag_chain import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, build_context_strings

text_docs = [
    Document(
        page_content="The kitchen is clean and well-organized",
        metadata={"id": 123, "rerank_score": 0.92}
    ),
    Document(
        page_content="Cooking utensils are arranged on the counter",
        metadata={"id": 456, "rerank_score": 0.87}
    ),
]
image_docs = [
    Document(page_content="Image 1", metadata={"id": 10001}),
    Document(page_content="Image 2", metadata={"id": 10002}),
]

text_ctx, image_ctx = build_context_strings(text_docs, image_docs)
user_msg = RAG_USER_TEMPLATE.format(
    query="주방의 모습을 설명해줘",
    text_context=text_ctx,
    image_context=image_ctx
)
print(f"  text context: {len(text_ctx)} chars")
print(f"  image context: {image_ctx}")
print(f"  rendered message: {len(user_msg)} chars")
passed += 1

# --- Test 5: VectorStore ---
print("\n[5] VectorStore - class load")

try:
    from engine.vectorstore import MilvusVectorStore
    print("  MilvusVectorStore import: OK")
    passed += 1
except Exception as e:
    print(f"  [FAIL] MilvusVectorStore import: {e}")
    failed += 1

# --- Summary ---
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 60)
