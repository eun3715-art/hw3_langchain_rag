#!/usr/bin/env python3
"""
RAG 시스템 통합 테스트
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("test_rag_system")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("RAG system integration test")
print("=" * 60)

# --- Test 1: Environment ---
print("\n[1] Environment check")

numpy_ok = False

try:
    import torch
    print(f"  torch: {torch.__version__}")
except ImportError as e:
    print(f"  torch: NOT FOUND ({e})")

try:
    import numpy as np
    print(f"  numpy: {np.__version__}")
    numpy_ok = True
except ImportError as e:
    print(f"  numpy: NOT FOUND ({e})")

try:
    import pandas as pd
    print(f"  pandas: {pd.__version__}")
except ImportError as e:
    print(f"  pandas: NOT FOUND ({e})")

try:
    from pymilvus import MilvusClient
    print("  pymilvus: OK")
except ImportError as e:
    print(f"  pymilvus: NOT FOUND ({e})")

try:
    from groq import Groq
    print("  groq: OK")
except ImportError as e:
    print(f"  groq: NOT FOUND ({e})")

try:
    from langchain_core.embeddings import Embeddings
    print("  langchain_core: OK")
except ImportError as e:
    print(f"  langchain_core: NOT FOUND ({e})")

try:
    from fastapi import FastAPI
    print("  fastapi: OK")
except ImportError as e:
    print(f"  fastapi: NOT FOUND ({e})")

# --- Test 2: Data files ---
print("\n[2] Data files")

DATA_ROOT = PROJECT_ROOT / "data"
print(f"  data root: {DATA_ROOT}")

txt_emb_file = DATA_ROOT / "txt_embeddings.npy"
img_emb_file = DATA_ROOT / "img_embeddings.npy"

if numpy_ok:
    if txt_emb_file.exists():
        txt_embeddings = np.load(txt_emb_file)
        print(f"  txt_embeddings.npy: {txt_embeddings.shape}")
    else:
        print("  txt_embeddings.npy: NOT FOUND")

    if img_emb_file.exists():
        img_embeddings = np.load(img_emb_file)
        print(f"  img_embeddings.npy: {img_embeddings.shape}")
    else:
        print("  img_embeddings.npy: NOT FOUND")
else:
    print("  embedding files: skipped (numpy not available)")

text_dir = DATA_ROOT / "texts"
if text_dir.exists():
    text_files = list(text_dir.glob("*.txt"))
    print(f"  text files: {len(text_files)}")
    if text_files:
        with open(text_files[0], "r", encoding="utf-8") as f:
            sample = f.read()
        print(f"  sample: {sample[:80]}...")
else:
    print("  texts/: NOT FOUND")

# --- Test 3: Router ---
print("\n[3] Hybrid Query Router")

try:
    from engine.router import route_query, init_sqlite_db
    init_sqlite_db()
    print("  SQLite DB init: OK")

    test_queries = [
        ("음식 준비 장면이 있는 이미지를 찾아줘", "vector"),
        ("문서의 평균 평점은?", "sql"),
        ("사람들이 함께 식사하는 장면", "vector"),
    ]

    for query, expected in test_queries:
        result = route_query(query, use_llm=False)
        route = result.get("route", "unknown")
        status = "PASS" if route == expected else "FAIL"
        print(f"  [{status}] '{query[:35]}' -> {route}")
        if route == "sql":
            print(f"         SQL: {result.get('sql', '')[:60]}")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# --- Test 4: Re-ranker ---
print("\n[4] Re-ranker")

try:
    from langchain_core.documents import Document
    from engine.reranker import turbo_rerank

    test_docs = [
        Document(page_content="Doc 1", metadata={"id": 1, "similarity_score": 0.80, "category": "A", "author": "X", "year": 2020}),
        Document(page_content="Doc 2", metadata={"id": 2, "similarity_score": 0.75, "category": "A", "author": "X", "year": 2020}),
        Document(page_content="Doc 3", metadata={"id": 3, "similarity_score": 0.82, "category": "B", "author": "Y", "year": 2021}),
    ]

    print("  Before:")
    for doc in test_docs:
        print(f"    ID={doc.metadata['id']}, score={doc.metadata['similarity_score']}")

    reranked = turbo_rerank(test_docs, alpha=0.15)

    print("  After:")
    for doc in reranked:
        print(f"    ID={doc.metadata['id']}, "
              f"base={doc.metadata.get('similarity_score', 0):.4f}, "
              f"bonus={doc.metadata.get('graphlet_bonus', 0):.4f}, "
              f"final={doc.metadata.get('rerank_score', 0):.4f}")
    print("  PASS")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# --- Test 5: Ingestion ---
print("\n[5] Ingestion pipeline")

try:
    from engine.ingestion import _compute_md5, _load_hash_registry

    test_file = PROJECT_ROOT / "test_sample.txt"
    test_file.write_text("Test content", encoding="utf-8")

    file_hash = _compute_md5(test_file)
    print(f"  MD5: {file_hash[:8]}...")

    registry = _load_hash_registry()
    print(f"  Registry: {len(registry)} entries")

    test_file.unlink()
    print("  PASS")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# --- Test 6: RAG Chain ---
print("\n[6] RAG Chain templates")

try:
    from engine.rag_chain import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

    rendered = RAG_USER_TEMPLATE.format(
        query="test query",
        text_context="test context",
        image_context="Image IDs: 1, 2, 3"
    )
    print(f"  System prompt: {len(RAG_SYSTEM_PROMPT)} chars")
    print(f"  Rendered template: {len(rendered)} chars")
    print("  PASS")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# --- Test 7: FastAPI routes ---
print("\n[7] FastAPI endpoint structure")

try:
    from main import app, AskRequest, AskResponse, IngestResponse, HealthResponse

    print(f"  AskRequest fields: {list(AskRequest.__fields__.keys())}")
    print(f"  AskResponse fields: {list(AskResponse.__fields__.keys())}")

    routes = [r for r in app.routes if hasattr(r, "path") and hasattr(r, "methods")]
    print(f"  Routes ({len(routes)}):")
    for r in routes:
        print(f"    {r.methods} {r.path}")
    print("  PASS")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# --- Test 8: CLIP embeddings ---
print("\n[8] CLIP embeddings (optional)")

try:
    from engine.embeddings import CLIPEmbeddings

    print("  Loading CLIP model...")
    clip = CLIPEmbeddings()
    embedding = clip.embed_query("A cat sitting on a table")
    print(f"  Embedding dim: {len(embedding)}")
    print(f"  Sample values: {[round(v, 4) for v in embedding[:3]]}")
    print("  PASS")

except Exception as e:
    print(f"  SKIP ({str(e)[:60]})")

print("\n" + "=" * 60 + "\n")
