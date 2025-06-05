
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from workflow.tools import norm
from db.repository import enqueue_job, insert_url, read_url, update_job, update_url
from config.chroma_client import client

def prompt_bool(msg: str) -> bool:
    while True:
        ans = input(f"{msg} [y/n]: ").lower().strip()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("y 또는 n으로 입력해주세요.")

if __name__ == '__main__':
    framework_nm = input("프레임워크 이름을 입력해주세요: ").strip()
    new_url = input("프레임워크의 공식문서 URL을 입력해주세요: ").strip()
    collection_nm = norm(framework_nm)

    # 기존 URL 조회
    old_url = read_url(collection_nm)

    if not old_url:
        # 기존 URL이 없다면 → 새롭게 등록
        enqueue_job(collection_nm, new_url)
        insert_url(collection_nm, new_url)
        print(f"[등록 완료] '{framework_nm}'에 대한 URL이 새로 등록되었습니다.")
    elif old_url == new_url:
        print(f"[안내] 입력하신 URL은 기존과 동일합니다. 작업을 종료합니다.")
    else:
        # URL이 다르면 → 사용자에게 선택을 맡김
        print(f"[주의] 기존 URL: {old_url}")
        print(f"[입력] 새로운 URL: {new_url}")
        if prompt_bool("기존 URL을 새 URL로 교체하시겠습니까?"):
            update_job(collection_nm, new_url)
            update_url(collection_nm, new_url)
            client.delete_collection(name=collection_nm)
            print(f"[업데이트 완료] '{framework_nm}'의 URL이 갱신되었습니다.")
        else:
            print("[중단] URL 변경이 취소되었습니다.")
