import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),         # 파일 로그
            logging.StreamHandler(sys.stdout)       # 콘솔 출력
        ]
    )