# Multimodal RAG Engine

텍스트 + 이미지 멀티모달 RAG 시스템. CLIP 임베딩으로 텍스트/이미지를 동일한 벡터 공간에 매핑하고, Milvus에서 유사도 검색 후 Groq LLM으로 답변을 생성한다.

---

## 구조

```
hw3_rag/
├── main.py               # FastAPI 서버
├── develop.py            # 벡터 DB 초기 셋업 스크립트 (최초 1회 실행)
├── requirements.txt
│
├── engine/               # 핵심 로직
│   ├── embeddings.py     # CLIPEmbeddings (텍스트 -> 512차원 벡터)
│   ├── vectorstore.py    # MilvusVectorStore (LangChain VectorStore 래퍼)
│   ├── router.py         # 쿼리 라우팅 (vector / sql 분기)
│   ├── reranker.py       # Graphlet-aware re-ranker
│   ├── ingestion.py      # 파일 수집, MD5 중복 체크
│   └── rag_chain.py      # end-to-end RAG 파이프라인
│
├── tests/
│   ├── test_rag_core.py  # 핵심 로직 단위 테스트 (FastAPI 제외)
│   └── test_rag_system.py
│
├── data/
│   ├── texts/            # 텍스트 원본 (25,014개)
│   ├── images/           # 이미지 원본 (5,000개)
│   ├── txt_embeddings.npy
│   └── img_embeddings.npy
│
├── docs/
│   ├── architecture.md   # 파일별 역할 설명
│   └── test_report.md
│
└── example/              # Milvus DiskANN 예제
```

---

## 시작하기

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수

```bash
export GROQ_API_KEY=<your-key>
```

### 3. 벡터 DB 셋업 (최초 1회)

`data/` 디렉터리에 `txt_embeddings.npy`, `img_embeddings.npy`, `texts/`, `images/` 가 있어야 한다.

```bash
python develop.py
```

### 4. 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

---

## API

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/health` | 서버 및 Milvus 상태 |
| POST | `/ask` | RAG 질의응답 |
| POST | `/ingest/file` | 단일 .txt 파일 수집 |
| POST | `/ingest/directory` | 디렉터리 일괄 수집 (백그라운드) |

### 질의 예시

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "요리하는 장면을 찾아줘", "k": 5, "use_rerank": true}'
```

---

## 테스트

```bash
python tests/test_rag_core.py    # 핵심 로직 (LLM 호출 없음)
python tests/test_rag_system.py  # 통합 테스트
```

---

## 파이프라인 흐름

```
query
  -> router.py        : vector / sql 분기
  -> vectorstore.py   : 텍스트 + 이미지 유사도 검색
  -> reranker.py      : Graphlet-aware 재정렬
  -> rag_chain.py     : Groq LLM으로 최종 답변 생성
```
