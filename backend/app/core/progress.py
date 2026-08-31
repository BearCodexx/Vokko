import time
import threading
from typing import Dict, Any, Optional

# потокобезопасное хранилище текущего прогресса и результатов задач
_progress_lock = threading.Lock()
_progress_store: Dict[str, Dict[str, Any]] = {}

# установка текущего этапа и сообщения задачи
def set_task_progress(task_id: str, stage: int, total_stages: int, message: str, percent: Optional[int] = None):
    if not task_id:
        return
    with _progress_lock:
        calc_percent = percent if percent is not None else int((stage / max(1, total_stages)) * 100)
        _progress_store[task_id] = {
            "status": "processing",
            "stage": stage,
            "total_stages": total_stages,
            "message": message,
            "percent": calc_percent,
            "updated_at": time.time()
        }

# сохранение успешного завершения задачи с готовыми данными
def set_task_completed(task_id: str, result_data: Dict[str, Any]):
    if not task_id:
        return
    with _progress_lock:
        _progress_store[task_id] = {
            "status": "completed",
            "stage": 4,
            "total_stages": 4,
            "message": "Обработка успешно завершена",
            "percent": 100,
            "result": result_data,
            "updated_at": time.time()
        }

# фиксация сбоя при выполнении задачи
def set_task_failed(task_id: str, error_message: str):
    if not task_id:
        return
    with _progress_lock:
        _progress_store[task_id] = {
            "status": "error",
            "message": error_message,
            "error": error_message,
            "percent": 0,
            "updated_at": time.time()
        }

# получение актуального состояния задачи по ее идентификатору
def get_task_progress(task_id: str) -> Dict[str, Any]:
    with _progress_lock:
        return _progress_store.get(task_id, {
            "status": "pending",
            "stage": 1,
            "total_stages": 4,
            "message": "Подготовка данных",
            "percent": 5
        })

# очистка записи задачи после завершения
def clear_task_progress(task_id: str):
    with _progress_lock:
        if task_id in _progress_store:
            del _progress_store[task_id]

