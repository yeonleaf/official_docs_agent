import json
import os, pathlib
import sys
import time
import sqlite3
import datetime
import streamlit as st
import logging
from config.constants import SQLITE_DB_PATH
from functools import wraps

logger = logging.getLogger(__name__)

def get_connection():
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn
    except Exception as e:
        logger.exception("DB 연결 실패")
        raise

def safe_write_block(fn, max_retries=5):
    for attempt in range(max_retries):
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                result = fn(cursor)  # 트랜잭션 블록 함수 실행
                conn.commit()
                return result
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                wait_time = 0.2 * (2 ** attempt)
                logger.warning(f"[경고] DB 잠김: {wait_time:.2f}s 후 재시도 (시도 {attempt + 1})")
                time.sleep(wait_time)
            else:
                raise
    logger.error("[실패] DB 잠금으로 인해 트랜잭션 포기")
    return None

def transactional_write(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return safe_write_block(
            lambda cursor: fn(*args, **kwargs, cursor=cursor)
        )
    return wrapper

# custom datetime adapter
def adapt_date_iso(val):
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()

def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat()

def adapt_datetime_epoch(val):
    """Adapt datetime.datetime to Unix timestamp."""
    return int(val.timestamp())

sqlite3.register_adapter(datetime.date, adapt_date_iso)
sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_adapter(datetime.datetime, adapt_datetime_epoch)

def convert_date(val):
    """Convert ISO 8601 date to datetime.date object."""
    return datetime.date.fromisoformat(val.decode())

def convert_datetime(val):
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())

def convert_timestamp(val):
    """Convert Unix epoch timestamp to datetime.datetime object."""
    return datetime.datetime.fromtimestamp(int(val))

sqlite3.register_converter("date", convert_date)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)

# init
def init_db():
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            # jobs 테이블 생성
            cursor.execute("""
            create table if not exists jobs (
                job_id integer primary key autoincrement,
                collection text not null,
                url text not null,
                status text check(status in ('pending', 'running', 'done', 'failed')),
                created_at timestamp default current_timestamp,
                updated_at timestamp default current_timestamp
            )
            """)
            # requests 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                question_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            )
            """)
            # urls 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                url_id INTEGER PRIMARY KEY AUTOINCREMENT,
                url text not null,
                collection text not null,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.commit()
            logger.info("DB 초기화 완료")
    except:
        logger.exception("DB 초기화 중 예외 발생")
        raise

# insert
@transactional_write
def insert_request(job_id: str, question_id: str, *, cursor):
    cursor.execute("""
        INSERT INTO requests (job_id, question_id, created_at)
        VALUES (?, ?, ?)
        """, (job_id, question_id, datetime.datetime.now()))
    

def select_job(collection: str):
    collection = collection.lower()
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # pending 잡 중 1개 선택
            cursor.execute("""
            select job_id, collection, url from jobs
            where collection = ?
            """, (collection,))

            job = cursor.fetchone()
            logger.info(f"잡 조회 성공 : {collection}")
            return job 
    except:
        logger.exception(f"잡 조회 실패 : {collection}")
        raise

@transactional_write
def enqueue_request(collection: str, url: str, question_id: str, *, cursor):
    collection = collection.lower()
    
    # ① 이미 있는 job 찾기
    cursor.execute(
        "SELECT job_id FROM jobs WHERE collection = ?",
        (collection,)
    )
    row = cursor.fetchone()

    # ② 없으면 새로 INSERT
    if row is None:
        cursor.execute(
            """INSERT INTO jobs (collection, url, status, created_at, updated_at)
            VALUES (?, ?, 'pending', ?, ?)""",
            (collection, url, datetime.datetime.now(), datetime.datetime.now())
        )
        job_id = cursor.lastrowid
    else:
        job_id = row[0]
        # 같은 질문에 request가 두 개 이상 들어가지 않도록 방지
        cursor.execute("""
        select * from requests where question_id = ? and job_id = ?
        """, (question_id, job_id))
        duplicated_request = cursor.fetchone()
        if duplicated_request:
            return
        
    # ③ request INSERT
    cursor.execute(
        """INSERT INTO requests (job_id, question_id, created_at)
        VALUES (?, ?, ?)""",
        (job_id, question_id, datetime.datetime.now())
    )

    logger.info(f"요청 큐 등록 성공 : {collection} {url} {question_id}")

def select_requests_by_question_id_and_status(question_id, status_list):
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # status_list 길이만큼 ? 플레이스홀더 생성
            placeholders = ','.join('?' for _ in status_list)
            query = f"""
            SELECT j.collection, j.status, r.created_at
            FROM requests AS r
            JOIN jobs AS j ON r.job_id = j.job_id AND j.status IN ({placeholders})
            WHERE r.question_id = ?
            ORDER BY j.status ASC, r.created_at ASC
            """
            
            # status_list + [question_id]를 파라미터로 전달
            cursor.execute(query, (*status_list, question_id))
            requests = cursor.fetchall()
            logger.info(f"요청 조회 성공 : {question_id} {",".join(status_list)}")
            return requests
    except:
        logger.exception(f"요청 조회 실패 : {question_id} {",".join(status_list)}")

@transactional_write
def enqueue_job(collection, url, *, cursor):
    collection = collection.lower()
    cursor.execute("""
    INSERT INTO jobs (collection, url, status, created_at, updated_at)
    VALUES (?, ?, 'pending', ?, ?)
    """, (collection, url, datetime.datetime.now(), datetime.datetime.now()))
    logger.info(f"잡 큐 등록 성공 : {collection} / {url}")

@transactional_write
def insert_url(collection: str, url: str, *, cursor):
    cursor.execute("""
    INSERT INTO urls (url, collection, created_at, updated_at)
    VALUES (?, ?, ?, ?)
    """, (collection, url, datetime.datetime.now(), datetime.datetime.now()))
    logger.info(f"URL 등록 성공 : {collection} / {url}")

# read
def read_queue_by_status(status: str, question_id_list: list):
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            base_query = """
                SELECT job_id, collection, url FROM jobs
                WHERE status = ?
                order by created_at ASC 
            """
            cursor.execute(base_query, (status,))
            jobs = cursor.fetchall()
            logger.info(f"잡 조회 성공 : {status} / {",".join(question_id_list)}")
            return jobs if jobs else []
    except:
        logger.exception(f"잡 조회 실패 : {status} / {",".join(question_id_list)}")

def read_url(collection: str):
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            base_query = """
                SELECT url FROM jobs
                WHERE collection = ?
                order by created_at ASC 
            """
            cursor.execute(base_query, (collection,))
            url = cursor.fetchone()
            logger.info(f"URL 조회 성공 : {collection} / {url}")
            return url[0] if url else None
    except:
        logger.exception(f"URL 조회 실패 : {collection}")

# update
@transactional_write
def update_job_status(job_id, status, *, cursor):
    cursor.execute("""
        update jobs set status=?, updated_at=? where job_id=?
    """, (status, datetime.now(), job_id))
    
@transactional_write
def update_job(collection, url, *, cursor):
    base_query = """
        update jobs
        set url = ?, updated_at = ?
        WHERE collection = ?
    """
    cursor.execute(base_query, (url, datetime.datetime.now(), collection))
    logger.info(f"잡 수정 성공 : {collection} / {url}")

@transactional_write
def update_url(collection, url, *, cursor):
    base_query = """
        update urls
        set url = ?, updated_at = ?
        WHERE collection = ?
    """
    cursor.execute(base_query, (url, datetime.datetime.now(), collection))
    logger.info(f"URL 수정 성공 : {collection} / {url}")