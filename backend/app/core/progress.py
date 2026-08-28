import threading
from typing import Dict, Any, Optional

# потокобезопасное хранилище текущего прогресса задач
_progress_lock = threading.Lock()
_progress_store: Dict[str, Dict[str, Any]] = {}

# установка текущего этапа и сообщения задачи
def set_task_progress(task_id: str, stage: int, total_stages: int, message: str):
    if not task_id:
        return
    with _progress_lock:
        percent = int((stage / max(1, total_stages)) * 100)
        _progress_store[task_id] = {
            "stage": stage,
            "total_stages": total_stages,
            "message": message,
            "percent": percent
        }

# получение состояния задачи по ее идентификатору
def get_task_progress(task_id: str) -> Dict[str, Any]:
    with _progress_lock:
        return _progress_store.get(task_id, {
            "stage": 1,
            "total_stages": 3,
            "message": "Подготовка данных",
            "percent": 10
        })

# очистка записи задачи после завершения
def clear_task_progress(task_id: str):
    with _progress_lock:
        if task_id in _progress_store:
            del _progress_store[task_id]
