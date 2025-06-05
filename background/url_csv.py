from datetime import datetime
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from workflow.tools import norm
from db.repository import enqueue_job, insert_url, read_url, update_job, update_url
from config.chroma_client import client

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("url_csv.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent

    csv_path = current_dir / 'urls.csv'
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        collection = None
        url = None
        try:
            collection = row["collection"]
            url = row["url"]
        except Exception as e:
            logger.error("양식이 잘못되었습니다. collection, url 헤더를 사용해 주세요")
        
        collection_nm = norm(collection)
        old_url = read_url(collection_nm)

        if not old_url:
            # 기존 URL이 없다면 → 새롭게 등록
            enqueue_job(collection_nm, url)
            insert_url(collection_nm, url)
            logger.info(f"'{collection_nm}'에 대한 URL이 새로 등록되었습니다.")
        elif old_url == url:
            logger.info(f"{collection_nm}에 대해 입력하신 URL은 기존과 동일합니다.")
        else:
            # URL이 다르면 → 덮어쓰기
            logger.info(f"{collection_nm}의 기존 URL과 신규 URL이 다릅니다. 덮어쓰기합니다.")
            logger.info(f"기존 URL: {old_url}")
            logger.info(f"새로운 URL: {url}")
            update_job(collection_nm, url)
            update_url(collection_nm, url)
            client.delete_collection(name=collection_nm)
            logger.info(f"'{collection_nm}'의 URL이 갱신되었습니다.")