
import chromadb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.chroma_client import client

if __name__ == '__main__':
    for collection in client.list_collections():
        print(f"erase {collection}", flush=True)
        client.delete_collection(name=collection)
    print("[알림] 모든 컬렉션을 삭제했습니다.")