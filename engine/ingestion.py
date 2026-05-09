import hashlib
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

from pymilvus import MilvusClient

from engine.embeddings import CLIPEmbeddings

logger = logging.getLogger("rag_engine.ingestion")

VECTOR_DB_DIR = Path(__file__).parent.parent / "vector_db"
HASH_REGISTRY_PATH = VECTOR_DB_DIR / "ingested_hashes.pkl"


def _load_hash_registry() -> Dict[str, str]:
    if HASH_REGISTRY_PATH.exists():
        with open(HASH_REGISTRY_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_hash_registry(registry: Dict[str, str]) -> None:
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    with open(HASH_REGISTRY_PATH, "wb") as f:
        pickle.dump(registry, f)


def _compute_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_file(
    file_path: Path,
    embeddings_model: CLIPEmbeddings,
    milvus_client: MilvusClient,
    collection_name: str,
    hash_registry: Dict[str, str]
) -> bool:
    """
    단일 파일 수집.
    - MD5 중복 체크
    - 임베딩 실패 시 에러 로그 후 스킵

    Returns:
        True: 성공 / False: 중복 또는 실패
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("[Ingestion] 파일 없음: %s", file_path)
        return False

    try:
        file_hash = _compute_md5(file_path)
    except Exception as e:
        logger.error("[Ingestion] 해시 계산 실패 (%s): %s", file_path.name, e)
        return False

    if file_hash in hash_registry:
        logger.info("[Ingestion] 중복 스킵: %s (hash=%s)", file_path.name, file_hash[:8])
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            logger.warning("[Ingestion] 빈 파일 스킵: %s", file_path.name)
            return False
        vector = embeddings_model.embed_query(content)
    except UnicodeDecodeError as e:
        logger.error("[Ingestion] 인코딩 오류 (%s): %s", file_path.name, e)
        return False
    except Exception as e:
        logger.error("[Ingestion] 임베딩 실패 (%s): %s", file_path.name, e)
        return False

    try:
        new_id = int(file_hash[:8], 16)
        milvus_client.insert(
            collection_name=collection_name,
            data=[{
                "id": new_id,
                "text_vector": vector,
                "text_content": content,
                "source_file": file_path.name,
                "ingested_at": datetime.now().isoformat()
            }]
        )
        hash_registry[file_hash] = str(file_path)
        _save_hash_registry(hash_registry)
        logger.info("[Ingestion] 수집 완료: %s (id=%d)", file_path.name, new_id)
        return True
    except Exception as e:
        logger.error("[Ingestion] DB 삽입 실패 (%s): %s", file_path.name, e)
        return False


def ingest_directory(
    directory: Path,
    embeddings_model: CLIPEmbeddings,
    milvus_client: MilvusClient,
    collection_name: str,
    pattern: str = "*.txt"
) -> Dict[str, int]:
    """
    디렉터리 내 파일 일괄 수집.

    Returns:
        {"success": int, "skipped": int, "failed": int}
    """
    directory = Path(directory)
    hash_registry = _load_hash_registry()
    stats = {"success": 0, "skipped": 0, "failed": 0}
    files = list(directory.glob(pattern))
    logger.info("[Ingestion] %d개 파일 수집 시작: %s", len(files), directory)

    for file_path in files:
        file_hash = None
        try:
            file_hash = _compute_md5(file_path)
        except Exception:
            pass

        result = ingest_file(
            file_path, embeddings_model, milvus_client, collection_name, hash_registry
        )
        if result:
            stats["success"] += 1
        elif file_hash and file_hash in hash_registry:
            stats["skipped"] += 1
        else:
            stats["failed"] += 1

    logger.info(
        "[Ingestion] 완료 | 성공: %d | 중복스킵: %d | 실패: %d",
        stats["success"], stats["skipped"], stats["failed"]
    )
    return stats
