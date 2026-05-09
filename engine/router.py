import json
import os
import re
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any

import pandas as pd

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger("rag_engine.router")

SQLITE_DB_PATH = Path(__file__).parent.parent / "structured_data.db"

ROUTER_SYSTEM_PROMPT = """당신은 RAG 시스템의 쿼리 라우터입니다.
사용자 질문을 분석하여 아래 두 가지 경로 중 하나만 선택하세요.

- "vector"  : 의미론적 유사도 검색이 필요한 경우 (내용 설명, 장면 묘사, 자연어 질의)
- "sql"     : 수치 집계, 통계, 필터링, 카운트가 필요한 경우 (평균, 최고, 몇 개, 년도별 등)

반드시 JSON 형태로만 응답하세요:
{"route": "vector"} 또는 {"route": "sql", "sql": "<실행할 SQL 쿼리>"}
테이블명은 'documents', 컬럼: id, category, author, year, word_count, rating"""

SQL_KEYWORDS = [
    "평균", "최고", "최저", "몇 개", "개수", "카운트", "합계", "통계",
    "년도", "몇 건", "average", "count", "sum", "max", "min",
    "how many", "statistics", "total"
]


def init_sqlite_db(db_path: Path = SQLITE_DB_PATH) -> None:
    """테스트용 SQLite DB 초기화 (이미 있으면 스킵)"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY,
            category    TEXT,
            author      TEXT,
            year        INTEGER,
            word_count  INTEGER,
            rating      REAL
        )
    """)
    sample_rows = [
        (i, f"cat_{i % 5}", f"author_{i % 20}", 2018 + (i % 6),
         200 + i * 3, round(3.0 + (i % 20) * 0.1, 1))
        for i in range(100)
    ]
    cur.executemany("INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?)", sample_rows)
    conn.commit()
    conn.close()
    logger.info("SQLite DB 초기화 완료: %s", db_path)


def _generate_fallback_sql(query: str) -> str:
    q = query.lower()
    if "평균" in q or "average" in q:
        return "SELECT category, AVG(rating) as avg_rating FROM documents GROUP BY category"
    if "최고" in q or "max" in q:
        return "SELECT * FROM documents ORDER BY rating DESC LIMIT 5"
    if "몇 개" in q or "count" in q or "how many" in q:
        return "SELECT category, COUNT(*) as cnt FROM documents GROUP BY category"
    return "SELECT * FROM documents LIMIT 10"


def route_query(query: str, use_llm: bool = True) -> Dict[str, str]:
    """
    LLM으로 쿼리 의도를 분석해 라우팅 결정.
    use_llm=False 이거나 LLM 호출 실패 시 규칙 기반 폴백 사용.
    """
    if use_llm and Groq is not None:
        try:
            client_llm = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            response = client_llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=200,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ]
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning("LLM 라우팅 실패, 폴백 사용: %s", e)

    # LLM 미사용 또는 실패 시 키워드 기반 폴백
    if any(kw in query.lower() for kw in SQL_KEYWORDS):
        sql = _generate_fallback_sql(query)
        return {"route": "sql", "sql": sql}

    return {"route": "vector"}


def execute_sql_query(sql: str, db_path: Path = SQLITE_DB_PATH) -> pd.DataFrame:
    """SQLite에서 SQL 실행 후 DataFrame 반환"""
    if not sql.strip().upper().startswith("SELECT"):
        logger.error("SELECT 외 쿼리 차단: %s", sql)
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
        logger.info("SQL 실행 성공: %s", sql)
        return df
    except Exception as e:
        logger.error("SQL 실행 실패: %s | 쿼리: %s", e, sql)
        return pd.DataFrame()
    finally:
        conn.close()
