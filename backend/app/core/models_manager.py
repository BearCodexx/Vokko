import os
import json
import urllib.request
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from backend.app.core.config import (
    CONFIG_FILE,
    MODELS_WHISPER_DIR,
    MODELS_LLM_DIR
)
from backend.app.core.logger import log_info, log_error

# каталог поддерживаемых моделей распознавания речи с объективными характеристиками
WHISPER_MODELS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "id": "tiny",
        "name": "Whisper Tiny (39M)",
        "type": "asr",
        "category": "Быстрая",
        "params": "39M",
        "accuracy": "WER ~19%",
        "speed": "~32x RT",
        "vram": "1.0 ГБ",
        "ssd": "75 МБ",
        "quality": "Базовое",
        "filename": "tiny.pt",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26fde3f8dd4e9508d0b43d049d9a149a7e0d3c/tiny.pt"
    },
    "base": {
        "id": "base",
        "name": "Whisper Base (74M)",
        "type": "asr",
        "category": "Быстрая",
        "params": "74M",
        "accuracy": "WER ~14%",
        "speed": "~16x RT",
        "vram": "1.2 ГБ",
        "ssd": "145 МБ",
        "quality": "Хорошее",
        "filename": "base.pt",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
    },
    "small": {
        "id": "small",
        "name": "Whisper Small (244M)",
        "type": "asr",
        "category": "Оптимальная",
        "params": "244M",
        "accuracy": "WER ~9.5%",
        "speed": "~6x RT",
        "vram": "2.0 ГБ",
        "ssd": "465 МБ",
        "quality": "Выше среднего",
        "filename": "small.pt",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d7205242f44e090fe7c1ecd8001fa3729caab/small.pt"
    },
    "medium": {
        "id": "medium",
        "name": "Whisper Medium (769M)",
        "type": "asr",
        "category": "Точная",
        "params": "769M",
        "accuracy": "WER ~6.2%",
        "speed": "~2x RT",
        "vram": "4.5 ГБ",
        "ssd": "1.5 ГБ",
        "quality": "Высокое",
        "filename": "medium.pt",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/34547462f743b1233d162d3411524143a08d27819ba7ab5083f2a7d25680baabb/medium.pt"
    },
    "turbo": {
        "id": "turbo",
        "name": "Whisper Large v3 Turbo (809M)",
        "type": "asr",
        "category": "Рекомендуемая",
        "params": "809M",
        "accuracy": "WER ~4.8%",
        "speed": "~8x RT",
        "vram": "4.0 ГБ",
        "ssd": "1.6 ГБ",
        "quality": "Отличное",
        "filename": "large-v3-turbo.pt",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408c342d825d461d6ddbd99d22d21e90163da53d36ea8683849714d42/large-v3-turbo.pt"
    },
    "large-v3": {
        "id": "large-v3",
        "name": "Whisper Large v3 (1.55B)",
        "type": "asr",
        "category": "Максимальная",
        "params": "1.55B",
        "accuracy": "WER ~3.9% SOTA",
        "speed": "~1.5x RT",
        "vram": "6.0 ГБ",
        "ssd": "3.1 ГБ",
        "quality": "Максимальное",
        "filename": "large-v3.pt",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeead8ff5e44ba1b8633992/large-v3.pt"
    }
}

# каталог поддерживаемых языковых моделей коррекции текста с объективными бенчмарками
LLM_MODELS: Dict[str, Dict[str, Any]] = {
    "none": {
        "id": "none",
        "name": "Без нейрокоррекции",
        "type": "llm",
        "category": "Нативный Whisper",
        "accuracy": "Оригинал",
        "speed": "Мгновенно",
        "vram": "0 ГБ",
        "ssd": "0 МБ",
        "quality": "Как есть",
        "filename": "",
        "url": ""
    },
    "llama3.2-3b": {
        "id": "llama3.2-3b",
        "name": "Llama 3.2 3B (Meta)",
        "type": "llm",
        "category": "Компактная (3.2B)",
        "accuracy": "RuMMLU 64%",
        "speed": "~70 т/с",
        "vram": "2.2 ГБ",
        "ssd": "2.0 ГБ",
        "quality": "Очень низкое (3B)",
        "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    },
    "phi3.5-mini": {
        "id": "phi3.5-mini",
        "name": "Phi-3.5-mini 3.8B (Microsoft)",
        "type": "llm",
        "category": "Компактная (3.8B)",
        "accuracy": "MMLU 69%",
        "speed": "~55 т/с",
        "vram": "2.6 ГБ",
        "ssd": "2.4 ГБ",
        "quality": "Низкое (3.8B)",
        "filename": "phi-3.5-mini-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
    },
    "qwen2.5-7b": {
        "id": "qwen2.5-7b",
        "name": "Qwen 2.5 7B (Alibaba)",
        "type": "llm",
        "category": "Средняя (7.6B)",
        "accuracy": "RuMMLU 71%",
        "speed": "~50 т/с",
        "vram": "5.1 ГБ",
        "ssd": "4.7 ГБ",
        "quality": "Среднее (7B)",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
    },
    "llama3.1-8b": {
        "id": "llama3.1-8b",
        "name": "Llama 3.1 8B (Meta)",
        "type": "llm",
        "category": "Средняя (8.0B)",
        "accuracy": "RuMMLU 72%",
        "speed": "~48 т/с",
        "vram": "5.5 ГБ",
        "ssd": "4.9 ГБ",
        "quality": "Среднее (8B)",
        "filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    },
    "gemma2-9b": {
        "id": "gemma2-9b",
        "name": "Gemma 2 9B (Google)",
        "type": "llm",
        "category": "Средняя (9.2B)",
        "accuracy": "RuMMLU 74%",
        "speed": "~42 т/с",
        "vram": "6.2 ГБ",
        "ssd": "5.8 ГБ",
        "quality": "Среднее (9B)",
        "filename": "gemma-2-9b-it-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"
    },
    "mistral-nemo-12b": {
        "id": "mistral-nemo-12b",
        "name": "Mistral NeMo 12B (NVIDIA)",
        "type": "llm",
        "category": "Тяжелая (12.2B)",
        "accuracy": "MMLU 73%",
        "speed": "~35 т/с",
        "vram": "8.2 ГБ",
        "ssd": "7.5 ГБ",
        "quality": "Высокое (12B Pro)",
        "filename": "mistral-nemo-instruct-2407-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
    },
    "qwen2.5-14b": {
        "id": "qwen2.5-14b",
        "name": "Qwen 2.5 14B (Alibaba)",
        "type": "llm",
        "category": "Тяжелая (14.7B)",
        "accuracy": "RuMMLU 78%",
        "speed": "~28 т/с",
        "vram": "9.8 ГБ",
        "ssd": "9.0 ГБ",
        "quality": "Высокое (14B Pro)",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf"
    },
    "qwen2.5-32b": {
        "id": "qwen2.5-32b",
        "name": "Qwen 2.5 32B (Alibaba)",
        "type": "llm",
        "category": "Тяжелая (32.5B)",
        "accuracy": "RuMMLU 83% SOTA",
        "speed": "~15 т/с",
        "vram": "19.5 ГБ",
        "ssd": "19.0 ГБ",
        "quality": "Максимальное (32B SOTA)",
        "filename": "qwen2.5-32b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/resolve/main/qwen2.5-32b-instruct-q4_k_m.gguf"
    },
    "saiga-llama3.2-3b": {
        "id": "saiga-llama3.2-3b",
        "name": "Saiga Llama 3.2 3B [RU]",
        "type": "llm",
        "category": "Русский язык",
        "accuracy": "Морфология RU",
        "speed": "~70 т/с",
        "vram": "2.2 ГБ",
        "ssd": "2.0 ГБ",
        "quality": "Среднее для RU (3.2B)",
        "filename": "saiga-llama3-3b-q4_k.gguf",
        "url": "https://huggingface.co/IlyaGusev/saiga_llama3_3b_gguf/resolve/main/model-q4_k.gguf"
    },
    "saiga-llama3.1-8b": {
        "id": "saiga-llama3.1-8b",
        "name": "Saiga Llama 3.1 8B [RU]",
        "type": "llm",
        "category": "Русский язык",
        "accuracy": "Стихи и вычитка RU",
        "speed": "~48 т/с",
        "vram": "5.5 ГБ",
        "ssd": "4.9 ГБ",
        "quality": "Высокое для RU (8.0B)",
        "filename": "saiga-llama3-8b-q4_k.gguf",
        "url": "https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_k.gguf"
    },
    "saiga-gemma2-9b": {
        "id": "saiga-gemma2-9b",
        "name": "Saiga Gemma 2 9B [RU]",
        "type": "llm",
        "category": "Русский язык",
        "accuracy": "Поэзия и смысл RU",
        "speed": "~42 т/с",
        "vram": "6.2 ГБ",
        "ssd": "5.8 ГБ",
        "quality": "Высокое для RU (9.2B)",
        "filename": "saiga-gemma2-9b-q4_k.gguf",
        "url": "https://huggingface.co/IlyaGusev/saiga_gemma2_9b_gguf/resolve/main/model-q4_k.gguf"
    },
    "openrouter": {
        "id": "openrouter",
        "name": "OpenRouter (Облачный API)",
        "type": "llm",
        "category": "Облако (OpenRouter)",
        "accuracy": "Флагманские модели",
        "speed": "По сети",
        "vram": "0 ГБ",
        "ssd": "0 МБ",
        "quality": "Облачный сервис",
        "filename": "",
        "url": ""
    },
    "custom-ollama": {
        "id": "custom-ollama",
        "name": "Ollama / Локальный сервер",
        "type": "llm",
        "category": "Локальный сервер (Ollama)",
        "accuracy": "Зависит от LLM",
        "speed": "Локально",
        "vram": "0 ГБ",
        "ssd": "0 МБ",
        "quality": "Локальный сервис",
        "filename": "",
        "url": ""
    }
}

# отслеживание прогресса загрузки моделей в фоновом режиме
_download_progress: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

# чтение текущих настроек пользователя из файла
def load_app_config() -> Dict[str, Any]:
    default_cfg = {
        "active_asr_model": "turbo",
        "active_llm_model": "llama3.1-8b",
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "llama3.1:8b",
        "openrouter_api_key": "",
        "openrouter_model": "google/gemma-2-9b-it",
        "wizard_completed": False
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception:
            pass
    return default_cfg

# сохранение настроек пользователя в файл
def save_app_config(cfg: Dict[str, Any]):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_error(f"Не удалось сохранить настройки: {str(e)}")

# проверка наличия скачанного файла модели на диске
def is_model_downloaded(model_id: str) -> bool:
    if model_id in ["none", "custom-ollama", "openrouter"]:
        return True

    if model_id in WHISPER_MODELS:
        m = WHISPER_MODELS[model_id]
        fn = m["filename"]
        target_in_local = MODELS_WHISPER_DIR / fn
        user_cache = Path.home() / ".cache" / "whisper" / fn
        return target_in_local.exists() or user_cache.exists()

    if model_id in LLM_MODELS:
        m = LLM_MODELS[model_id]
        fn = m["filename"]
        if not fn:
            return True
        return (MODELS_LLM_DIR / fn).exists()

    return False

# получение абсолютного пути к файлу модели
def get_model_path(model_id: str) -> Optional[Path]:
    if model_id in WHISPER_MODELS:
        fn = WHISPER_MODELS[model_id]["filename"]
        target_in_local = MODELS_WHISPER_DIR / fn
        if target_in_local.exists():
            return target_in_local
        user_cache = Path.home() / ".cache" / "whisper" / fn
        if user_cache.exists():
            return user_cache
        return target_in_local

    if model_id in LLM_MODELS:
        fn = LLM_MODELS[model_id]["filename"]
        if fn:
            return MODELS_LLM_DIR / fn

    return None

# получение текущего прогресса загрузки
def get_download_status(model_id: str) -> Dict[str, Any]:
    with _lock:
        if model_id in _download_progress:
            return _download_progress[model_id]
        downloaded = is_model_downloaded(model_id)
        return {
            "status": "ready" if downloaded else "not_downloaded",
            "percent": 100 if downloaded else 0,
            "error": None
        }

# фоновая загрузка файла модели с обновлением прогресса
def start_model_download(model_id: str):
    with _lock:
        status = _download_progress.get(model_id, {}).get("status")
        if status == "downloading":
            return

        _download_progress[model_id] = {
            "status": "downloading",
            "percent": 0,
            "error": None
        }

    thread = threading.Thread(target=_download_worker, args=(model_id,), daemon=True)
    thread.start()

# рабочий процесс скачивания файла по сети
def _download_worker(model_id: str):
    meta = WHISPER_MODELS.get(model_id) or LLM_MODELS.get(model_id)
    if not meta or not meta.get("url"):
        with _lock:
            _download_progress[model_id] = {"status": "error", "percent": 0, "error": "Ссылка на модель не найдена"}
        return

    url = meta["url"]
    fn = meta["filename"]
    target_dir = MODELS_WHISPER_DIR if meta["type"] == "asr" else MODELS_LLM_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / fn
    temp_file = target_dir / f"{fn}.part"

    try:
        log_info(f"Начало загрузки модели {model_id} из {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024

            with open(temp_file, "wb") as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int((downloaded / total_size) * 100)
                        with _lock:
                            _download_progress[model_id]["percent"] = min(99, pct)

        if temp_file.exists():
            if target_file.exists():
                target_file.unlink()
            temp_file.rename(target_file)

        with _lock:
            _download_progress[model_id] = {"status": "ready", "percent": 100, "error": None}
        log_info(f"Модель {model_id} успешно загружена и сохранена в {target_file}")

    except Exception as e:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        err_msg = str(e)
        log_error(f"Ошибка загрузки модели {model_id}: {err_msg}")
        with _lock:
            _download_progress[model_id] = {"status": "error", "percent": 0, "error": err_msg}

# удаление скачанного файла модели с диска
def delete_model_file(model_id: str) -> bool:
    path = get_model_path(model_id)
    if not path or not path.exists():
        return False
    try:
        path.unlink()
        with _lock:
            if model_id in _download_progress:
                del _download_progress[model_id]
        log_info(f"Файл модели {model_id} удален с диска")
        return True
    except Exception as e:
        log_error(f"Не удалось удалить файл модели {model_id}: {str(e)}")
        return False

# получение полного каталога моделей со статусом загрузки
def get_all_models_catalog() -> Dict[str, Any]:
    cfg = load_app_config()
    asr_list = []
    for mid, m in WHISPER_MODELS.items():
        item = dict(m)
        item["downloaded"] = is_model_downloaded(mid)
        asr_list.append(item)

    llm_list = []
    for mid, m in LLM_MODELS.items():
        item = dict(m)
        item["downloaded"] = is_model_downloaded(mid)
        llm_list.append(item)

    return {
        "active_asr": cfg.get("active_asr_model", "turbo"),
        "active_llm": cfg.get("active_llm_model", "llama3.1-8b"),
        "ollama_url": cfg.get("ollama_url", "http://127.0.0.1:11434"),
        "ollama_model": cfg.get("ollama_model", "llama3.1:8b"),
        "openrouter_api_key": cfg.get("openrouter_api_key", ""),
        "openrouter_model": cfg.get("openrouter_model", "google/gemma-2-9b-it"),
        "asr_models": asr_list,
        "llm_models": llm_list
    }

download_model_file = start_model_download
get_download_progress = get_download_status
