import os
import subprocess
import signal
from pathlib import Path

if __name__ == "__main__":
    current_path = Path(__file__).parent

    # worker 프로세스 내리기
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "worker.py" in line and "grep" not in line:
            try:
                parts = line.split()
                pid = int(parts[1])  # 두 번째 항목이 PID
                print(f"worker.py 프로세스를 종료합니다. PID: {pid}")
                os.kill(pid, signal.SIGTERM)
                break
            except Exception as e:
                print(f"오류 발생: {e}")

    # sqlite db 삭제
    for name in ["jobs.db", "jobs.db-shm", "jobs.db-wal"]:
        path = current_path / "db" / name
        if path.exists():
            path.unlink()
            print(f"{name} 삭제 완료")

    # 컬렉션 비우기
    subprocess.run(["python", str(current_path / "background" / "erase.py")])

    # 로그 삭제
    for log in ["app.log", "background/worker.log"]:
        log_path = current_path / log
        if log_path.exists():
            log_path.unlink()
            print(f"{log} 삭제 완료")

    # worker 프로세스 올리기 (nohup + &를 shell로 처리)
    worker_path = current_path / "background" / "worker.py"
    worker_log_path = current_path / "background" / "worker.log"
    subprocess.run(
        f"nohup python {worker_path} > {worker_log_path} 2>&1 &",
        shell=True
    )
    print("worker.py 재시작 완료")