import os
import logging
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from langchain_core.documents import Document

from engine.reranker import turbo_rerank
from engine.router import route_query, execute_sql_query

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger("rag_engine.rag_chain")

RAG_SYSTEM_PROMPT = """당신은 멀티모달 RAG 시스템의 답변 생성기입니다.
아래 규칙을 반드시 따르세요:

1. 제공된 텍스트 컨텍스트와 이미지 ID를 근거로만 답변합니다.
2. 모든 답변 마지막에 반드시 다음 형식으로 출처를 명시합니다:

---
참조 출처:
  - 텍스트: [ID 목록 및 핵심 내용 요약]
  - 이미지: [Image ID 목록]
---

3. 컨텍스트에 없는 내용은 "제공된 데이터에 없는 정보입니다"라고 답합니다.
4. 한국어로 답변합니다."""

RAG_USER_TEMPLATE = """
[질문]
{query}

[검색된 텍스트 컨텍스트]
{text_context}

[검색된 이미지 ID]
{image_context}

위 컨텍스트를 바탕으로 질문에 답변하고, 반드시 출처를 명시하세요.
"""


def build_context_strings(
    text_docs: List[Document],
    image_docs: List[Document]
) -> Tuple[str, str]:
    text_parts = []
    for i, doc in enumerate(text_docs, 1):
        doc_id = doc.metadata.get("id", "?")
        score = doc.metadata.get("rerank_score", doc.metadata.get("similarity_score", 0.0))
        text_parts.append(
            f"[텍스트 #{i} | ID={doc_id} | score={score:.4f}]\n{doc.page_content}"
        )
    text_context = "\n\n".join(text_parts) if text_parts else "검색 결과 없음"

    image_ids = [str(doc.metadata.get("id", "?")) for doc in image_docs]
    image_context = "Image IDs: " + ", ".join(image_ids) if image_ids else "검색 결과 없음"
    return text_context, image_context


def rag_answer(
    query: str,
    text_vectorstore,
    image_vectorstore,
    k: int = 5,
    use_rerank: bool = True
) -> Dict[str, Any]:
    """
    End-to-end RAG 파이프라인.
    Router -> 검색 -> (Re-rank) -> LLM 답변 생성
    """
    routing = route_query(query)
    route = routing.get("route", "vector")

    if route == "sql":
        sql = routing.get("sql", "SELECT * FROM documents LIMIT 10")
        df = execute_sql_query(sql)
        answer = f"[SQL 결과]\n{df.to_string(index=False)}"
        return {
            "answer": answer,
            "route": "sql",
            "text_sources": [],
            "image_sources": [],
            "sql_result": df
        }

    search_k = k * 2 if use_rerank else k
    text_docs = text_vectorstore.similarity_search(query, k=search_k)
    image_docs = image_vectorstore.similarity_search(query, k=search_k)

    if use_rerank:
        text_docs = turbo_rerank(text_docs)[:k]
        image_docs = turbo_rerank(image_docs)[:k]

    text_context, image_context = build_context_strings(text_docs, image_docs)

    if Groq is None:
        logger.error("[RAG] groq 패키지가 설치되지 않았습니다.")
        answer = f"groq 패키지가 필요합니다.\n\n[컨텍스트]\n{text_context}"
    else:
        try:
            client_llm = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            user_message = RAG_USER_TEMPLATE.format(
                query=query,
                text_context=text_context,
                image_context=image_context
            )
            response = client_llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ]
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error("[RAG] LLM 호출 실패: %s", e)
            answer = f"LLM 호출에 실패했습니다: {e}\n\n[컨텍스트]\n{text_context}"

    text_sources = [
        {
            "id": d.metadata.get("id"),
            "score": d.metadata.get("rerank_score", 0.0),
            "content_preview": d.page_content[:80]
        }
        for d in text_docs
    ]
    image_sources = [d.metadata.get("id") for d in image_docs]

    logger.info("[RAG] 답변 생성 완료 | route=%s | 쿼리: %s", route, query)
    return {
        "answer": answer,
        "route": route,
        "text_sources": text_sources,
        "image_sources": image_sources,
        "sql_result": None
    }
