import logging
from typing import List, Optional, Dict

from langchain_core.documents import Document

logger = logging.getLogger("rag_engine.reranker")


def turbo_rerank(
    documents: List[Document],
    alpha: float = 0.15,
    shared_keys: Optional[List[str]] = None
) -> List[Document]:
    """
    TurboLynx 식 Graphlet-aware 재순위 함수.

    Args:
        documents  : similarity_search 결과 (metadata에 similarity_score 포함)
        alpha      : 공유 메타데이터 1건당 보너스 가중치 (기본 0.15)
        shared_keys: 비교할 메타데이터 키 목록 (기본: category, author, year)

    Returns:
        재순위된 Document 리스트 (내림차순)
    """
    if shared_keys is None:
        shared_keys = ["category", "author", "year"]

    if len(documents) <= 1:
        return documents

    scored: List[Dict] = [
        {
            "doc": doc,
            "base_score": doc.metadata.get("similarity_score", 0.0),
            "graphlet_bonus": 0.0
        }
        for doc in documents
    ]

    # 문서 쌍 간 공통 메타데이터 카운트 → 보너스 부여
    for i in range(len(scored)):
        for j in range(len(scored)):
            if i == j:
                continue
            meta_i = scored[i]["doc"].metadata
            meta_j = scored[j]["doc"].metadata
            shared_count = sum(
                1 for k in shared_keys
                if meta_i.get(k) is not None and meta_i.get(k) == meta_j.get(k)
            )
            scored[i]["graphlet_bonus"] += alpha * shared_count

    for item in scored:
        item["final_score"] = item["base_score"] + item["graphlet_bonus"]
        item["doc"].metadata["rerank_score"] = round(item["final_score"], 4)
        item["doc"].metadata["graphlet_bonus"] = round(item["graphlet_bonus"], 4)

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    logger.info(
        "[Re-ranker] %d개 문서 재순위 완료 | alpha=%.2f | keys=%s",
        len(documents), alpha, shared_keys
    )
    return [item["doc"] for item in scored]
