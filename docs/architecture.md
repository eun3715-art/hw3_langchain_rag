# 프로젝트 구조 설명

```
hw3_rag/
├── engine/
│   ├── embeddings.py     # CLIPEmbeddings - 텍스트/이미지 -> 512차원 벡터
│   ├── vectorstore.py    # MilvusVectorStore - LangChain VectorStore 래퍼
│   ├── router.py         # 쿼리 라우팅 (vector vs sql), SQLite 연동
│   ├── reranker.py       # Graphlet-aware re-ranker
│   ├── ingestion.py      # 파일 수집, MD5 중복 체크
│   └── rag_chain.py      # end-to-end RAG 파이프라인
│
├── main.py               # FastAPI 서버
├── develop.py            # 최초 벡터 DB 셋업 스크립트
├── test_rag_core.py      # 핵심 로직 단위 테스트 (FastAPI 제외)
├── test_rag_system.py    # 통합 테스트
│
└── data/database/
    ├── txt_embeddings.npy   # 텍스트 임베딩 (25014, 512)
    ├── img_embeddings.npy   # 이미지 임베딩 (5000, 512)
    ├── texts/               # 텍스트 원본 파일
    └── images/              # 이미지 원본 파일
```

---

## 각 파일 설명

### `engine/embeddings.py`

`CLIPEmbeddings` 클래스. `openai/clip-vit-base-patch32` 모델로 텍스트를 512차원 벡터로 변환한다.
출력은 L2 정규화되어 내적(IP) 유사도 검색에 바로 사용할 수 있다.
LangChain `Embeddings` 인터페이스를 상속하므로 다른 LangChain 컴포넌트와 호환된다.

### `engine/vectorstore.py`

Milvus Lite를 LangChain `VectorStore` 인터페이스로 래핑한 `MilvusVectorStore` 클래스.
`similarity_search()` 와 `similarity_search_with_score()` 를 구현한다.
텍스트 컬렉션(`text_search`)과 이미지 컬렉션(`image_search`) 모두 이 클래스 하나로 처리한다.

### `engine/router.py`

사용자 쿼리를 분석해 `vector` 또는 `sql` 경로로 분기한다.

- 1차: Groq LLM으로 의도 분류 (JSON 응답 파싱)
- 2차 폴백: 키워드 기반 규칙 (`SQL_KEYWORDS` 리스트)
- LLM 없이 동작시키려면 `route_query(query, use_llm=False)` 로 호출

SQLite DB(`structured_data.db`)에 샘플 데이터가 있고, `execute_sql_query()`로 SELECT만 실행 가능하다.

### `engine/reranker.py`

`turbo_rerank()` 함수. 검색 결과 문서들 간 공통 메타데이터(category, author, year) 개수를 세어 보너스 점수를 추가한 뒤 재정렬한다.

```
final_score = similarity_score + alpha * shared_metadata_count
```

### `engine/ingestion.py`

파일을 벡터 DB에 수집한다. MD5 해시로 중복을 막고, 해시 레지스트리를 `vector_db/ingested_hashes.pkl`에 저장한다.

- `ingest_file()`: 단일 파일
- `ingest_directory()`: 디렉터리 일괄 수집

### `engine/rag_chain.py`

전체 RAG 파이프라인을 `rag_answer()` 함수 하나로 묶는다.

```
route_query() -> similarity_search() -> turbo_rerank() -> Groq LLM
```

SQL 경로인 경우 벡터 검색 없이 바로 `execute_sql_query()` 결과를 반환한다.

### `main.py`

FastAPI 서버. 엔드포인트:

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 및 Milvus 상태 확인 |
| POST | `/ask` | RAG 질의응답 |
| POST | `/ingest/file` | 단일 파일 업로드 수집 |
| POST | `/ingest/directory` | 디렉터리 일괄 수집 (백그라운드) |

### `develop.py`

최초 1회 실행하는 셋업 스크립트. `data/database/`의 npy 임베딩 파일을 읽어 Milvus 컬렉션을 생성하고 데이터를 삽입한다.

---

## 실행 순서

```bash
# 1. 벡터 DB 셋업 (최초 1회)
python develop.py

# 2. 서버 시작
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. API 테스트
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "요리하는 장면을 찾아줘"}'
```

## 환경 변수

| 변수 | 설명 |
|------|------|
| `GROQ_API_KEY` | Groq API 키. 미설정 시 LLM 호출 실패 (컨텍스트만 반환) |
