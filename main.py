"""
main.py - Multimodal RAG Engine FastAPI 서버
실행: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from engine.embeddings import CLIPEmbeddings
from engine.vectorstore import MilvusVectorStore
from engine.ingestion import ingest_file, ingest_directory, _load_hash_registry
from engine.router import init_sqlite_db
from engine.rag_chain import rag_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_engine.api")

COLLECTION_NAME_TEXT = "text_search"
COLLECTION_NAME_IMAGE = "image_search"
DATA_ROOT = Path(__file__).parent / "data"

logger.info("Milvus 클라이언트 초기화 중...")
milvus_client = MilvusClient(uri="./milvus_local.db")

logger.info("CLIP 모델 로드 중...")
clip_embeddings = CLIPEmbeddings()

text_vectorstore = MilvusVectorStore(
    client=milvus_client,
    collection_name=COLLECTION_NAME_TEXT,
    embeddings=clip_embeddings
)
image_vectorstore = MilvusVectorStore(
    client=milvus_client,
    collection_name=COLLECTION_NAME_IMAGE,
    embeddings=clip_embeddings
)

init_sqlite_db()
logger.info("서버 초기화 완료")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="사용자 질문")
    k: int = Field(default=5, ge=1, le=20, description="검색할 문서 수")
    use_rerank: bool = Field(default=True, description="Re-ranker 사용 여부")


class TextSource(BaseModel):
    id: Optional[int]
    score: float
    content_preview: str


class AskResponse(BaseModel):
    answer: str
    route: str
    text_sources: List[TextSource]
    image_sources: List[Optional[int]]
    sql_result: Optional[str] = None
    elapsed_ms: float


class IngestResponse(BaseModel):
    success: int
    skipped: int
    failed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool
    text_collection_size: int
    image_collection_size: int
    version: str


app = FastAPI(
    title="Multimodal RAG Engine API",
    description="텍스트 + 이미지 멀티모달 RAG 엔진 (Milvus + CLIP + Claude)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """서버 및 벡터 DB 상태 확인"""
    try:
        text_stats = milvus_client.get_collection_stats(COLLECTION_NAME_TEXT)
        image_stats = milvus_client.get_collection_stats(COLLECTION_NAME_IMAGE)
        text_count = text_stats.get("row_count", 0)
        image_count = image_stats.get("row_count", 0)
        milvus_ok = True
    except Exception as e:
        logger.error("[Health] Milvus 연결 실패: %s", e)
        text_count = image_count = 0
        milvus_ok = False

    return HealthResponse(
        status="ok" if milvus_ok else "degraded",
        milvus_connected=milvus_ok,
        text_collection_size=text_count,
        image_collection_size=image_count,
        version="1.0.0"
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest):
    """멀티모달 RAG 질의응답. 쿼리를 자동 라우팅(벡터/SQL) 후 Re-rank, LLM 답변 반환."""
    start = time.time()
    try:
        result = rag_answer(
            query=request.query,
            text_vectorstore=text_vectorstore,
            image_vectorstore=image_vectorstore,
            k=request.k,
            use_rerank=request.use_rerank
        )
    except Exception as e:
        logger.error("[API /ask] 처리 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = (time.time() - start) * 1000

    sql_json = None
    if result.get("sql_result") is not None:
        sql_json = result["sql_result"].to_json(orient="records", force_ascii=False)

    return AskResponse(
        answer=result["answer"],
        route=result["route"],
        text_sources=[TextSource(**s) for s in result["text_sources"]],
        image_sources=result["image_sources"],
        sql_result=sql_json,
        elapsed_ms=round(elapsed, 2)
    )


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file_endpoint(
    file: UploadFile = File(..., description="수집할 텍스트 파일 (.txt)")
):
    """단일 파일 업로드 후 벡터 DB에 수집 (MD5 중복 체크 포함)"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="현재 .txt 파일만 지원합니다.")

    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

    hash_registry = _load_hash_registry()
    success = ingest_file(
        tmp_path, clip_embeddings, milvus_client, COLLECTION_NAME_TEXT, hash_registry
    )
    tmp_path.unlink(missing_ok=True)

    if success:
        return IngestResponse(success=1, skipped=0, failed=0, message="파일 수집 완료")
    else:
        return IngestResponse(success=0, skipped=1, failed=0,
                              message="중복 파일이거나 수집에 실패했습니다.")


@app.post("/ingest/directory", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_directory_endpoint(
    background_tasks: BackgroundTasks,
    directory: str = str(DATA_ROOT / "texts"),
    pattern: str = "*.txt"
):
    """서버 내 디렉터리를 백그라운드에서 일괄 수집"""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"디렉터리 없음: {directory}")

    background_tasks.add_task(
        ingest_directory, dir_path, clip_embeddings, milvus_client, COLLECTION_NAME_TEXT, pattern
    )
    return IngestResponse(
        success=0, skipped=0, failed=0,
        message=f"백그라운드 수집 시작됨: {directory} (패턴: {pattern})"
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
