import os
import sys
import socket
import warnings
import logging
from pathlib import Path

# отключение системных предупреждений, тихий режим работы
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CT2_VERBOSE"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

# отключение лишнего вывода журналов в консоль
for logger_name in ["uvicorn", "uvicorn.access", "fastapi", "huggingface_hub", "demucs"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# базовые пути к каталогам проекта, временные файлы и модели
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
UPLOAD_DIR = BASE_DIR / "temp_storage" / "uploads"
PROCESSED_DIR = BASE_DIR / "temp_storage" / "processed"
EXPORTS_DIR = BASE_DIR / "temp_storage" / "exports"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_LLM_DIR = MODELS_DIR / "llm"
MODELS_WHISPER_DIR = MODELS_DIR / "whisper"
CONFIG_FILE = DATA_DIR / "config.json"
WIZARD_STATE_FILE = DATA_DIR / "wizard_state.json"

for d in [UPLOAD_DIR, PROCESSED_DIR, EXPORTS_DIR, DATA_DIR, MODELS_DIR, MODELS_LLM_DIR, MODELS_WHISPER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# поиск свободного сетевого порта, исключение конфликтов
def find_free_port(start_port: int = 8088) -> int:
    for p in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

# настройки параметров сервера, выбор модели
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
DEMUCS_MODEL_NAME = os.getenv("DEMUCS_MODEL", "htdemucs")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", str(find_free_port(8088))))
