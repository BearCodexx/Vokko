import os
import sys
import warnings
import logging
from contextlib import contextmanager

# установка переменных среды для отключения системных сообщений
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CT2_VERBOSE"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# полное отключение вывода предупреждений языка
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.showwarning = lambda *args, **kwargs: None

# подавление журналов сторонних библиотек
logging.disable(logging.WARNING)
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "huggingface_hub", "demucs", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# подавление вывода на уровне дескрипторов операционной системы
@contextmanager
def silence_stderr():
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        saved_stderr = os.dup(2)
        os.dup2(null_fd, 2)
        try:
            yield
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            os.close(null_fd)
    except Exception:
        yield
