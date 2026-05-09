# Multimodal RAG Engine - Test Report

**Date:** 2026-05-05  
**Environment:** macOS, Python 3.10

---

## Summary

5개 컴포넌트 기본 동작 검증 완료.

---

## Step 1: Hybrid Query Router (`engine/router.py`)

규칙 기반 SQL 키워드 감지, LLM 폴백, SQLite 실행 확인.

```
"음식 준비 사진을 보여줘"      -> vector  (pass)
"평균 평점이 높은 문서들은?"   -> sql     (pass)
"사람들이 웃고 있는 장면"      -> vector  (pass)
"카테고리별 문서 개수는?"      -> vector  (fail - "개수" 미등록)

SQL 실행: SELECT category, COUNT(*) FROM documents GROUP BY category
-> 5개 카테고리 x 20개 문서
```

참고: "개수" 키워드가 SQL_KEYWORDS 목록에 없어서 vector로 라우팅됨.

---

## Step 2: Graphlet-aware Re-ranker (`engine/reranker.py`)

similarity_score 기반 정렬 + 공통 메타데이터 보너스 적용 확인.

```
alpha=0.2 테스트:

Before: ID=1(0.85), ID=2(0.80), ID=3(0.75), ID=4(0.70), ID=5(0.65)
After:  ID=3(1.95), ID=1(1.85), ID=2(1.80), ID=4(1.50), ID=5(1.45)
```

ID=3이 공통 메타데이터(category, author, year) 6건으로 보너스 누적 -> 1위.

---

## Step 3: Ingestion Pipeline (`engine/ingestion.py`)

MD5 해시 기반 중복 체크, 레지스트리 영속성 확인.

```
file1.txt (2e71d7da...): 수집
file2.txt (20c1d191...): 수집
file3.txt (2e71d7da...): 중복 감지 -> 스킵

레지스트리: 저장 2개 -> 로드 2개 (pass)
```

---

## Step 4: RAG Chain (`engine/rag_chain.py`)

라우팅 분기, 병렬 검색, re-ranker 통합, 프롬프트 렌더링 확인.

```
텍스트 컨텍스트: 151자 (문서 2개)
이미지 컨텍스트: Image IDs: 10001, 10002
렌더링된 메시지: 264자
```

---

## Step 5: VectorStore + Embeddings (`engine/vectorstore.py`, `engine/embeddings.py`)

```
CLIP 모델:       openai/clip-vit-base-patch32
임베딩 차원:     512
정규화:          L2

텍스트 임베딩:   (25014, 512)
이미지 임베딩:   (5000, 512)
텍스트 파일:     25014개
```

---

## Known Issues

- `SQL_KEYWORDS`에 "개수" 미포함 -> "카테고리별 문서 개수는?" vector 라우팅
- MD5 앞 8자리로 Milvus ID 생성 (32-bit) -> 대용량 수집 시 충돌 가능성 있음
