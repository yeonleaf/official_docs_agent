import json
import os
import time
import sqlite3
from datetime import datetime
import asyncio

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from workflow.tools import crawl_and_embed
from db.repository import init_db, get_connection, update_job_status

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("worker.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def process_queue():
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # pending 잡 중 1개 선택
        cursor.execute("""
        select job_id, collection, url from jobs
        where status = 'pending'
        order by created_at asc
        limit 1
        """)
        job = cursor.fetchone()

        # 잡 처리
        if job:
            job_id, collection, url = job
            logger.info(f"[시작] {collection} / {url}", flush=True)
            update_job_status(job_id, 'running')
            conn.commit()
            try:
                asyncio.run(crawl_and_embed(collection, url))
                update_job_status(job_id, 'done')
            except Exception as e:
                update_job_status(job_id, 'failed')
                logger.error(f"{collection} / {url} 작업 실패: {e}", flush=True)
            conn.commit()
            logger.info(f"[종료] {collection} / {url}", flush=True)
        else:
            logger.info("[대기 중인 작업 없음]", flush=True)


if __name__ == '__main__':
    init_db()
    while True:
        logger.info("작업 시작", flush=True)
        process_queue()
        logger.info("작업 종료", flush=True)
        time.sleep(10)