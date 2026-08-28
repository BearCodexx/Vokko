import os
import re
import urllib.request
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.app.core.audio_utils import convert_to_wav
from backend.app.core.silence import silence_stderr
from backend.app.core.logger import log_info, log_progress, log_error
from backend.app.core.models_manager import (
    WHISPER_MODELS,
    get_model_path,
    download_model_file,
    is_model_downloaded,
    load_app_config
)

# модуль распознавания речи, преобразование звука в текст
class WhisperEngine:

    # шаблоны поиска мусорных фраз и авторов субтитров
    HALLUCINATION_PATTERNS = [
        r"dimatorzok",
        r"субтитр\w*",
        r"спасибо\b\.?",
        r"спасибо за просмотр",
        r"подписывайтесь",
        r"подпишитесь",
        r"ставьте лайк",
        r"до скор\w+ встреч\w*",
        r"всем пока",
        r"редактор субтитров",
        r"продолжение следует",
        r"перевод на русский",
        r"thank you",
        r"thanks for watching",
        r"subscribe",
        r"subtitles by"
    ]

    def __init__(self, model_size: Optional[str] = None):
        cfg = load_app_config()
        self.model_size = model_size or cfg.get("active_asr_model", "large-v3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._loaded_model_name = None

    # загрузка нейросетевой модели с проверкой наличия весов
    def _load_model(self, target_name: Optional[str] = None):
        model_name = target_name or self.model_size
        if self._model is not None and self._loaded_model_name == model_name:
            return self._model

        import whisper
        if not is_model_downloaded(model_name):
            log_info(f"Скачивание весов модели Whisper {model_name}")
            download_model_file(model_name)

        model_path = get_model_path(model_name)
        load_target = str(model_path) if model_path and model_path.exists() else model_name

        with silence_stderr():
            try:
                self._model = whisper.load_model(load_target, device=self.device)
                self._loaded_model_name = model_name
                log_info(f"Модель Whisper {model_name} готова на устройстве {self.device}")
                return self._model
            except Exception as e:
                log_error(f"Не удалось загрузить Whisper {model_name} на GPU, пробуем запасной вариант: {str(e)}")

            try:
                self._model = whisper.load_model("base", device=self.device)
                self._loaded_model_name = "base"
                return self._model
            except Exception:
                pass

            self._model = whisper.load_model("base", device="cpu")
            self._loaded_model_name = "base"
            return self._model

    # очистка строки от мусорных повторов, артефактов и циклов
    @classmethod
    def _clean_text(cls, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()

        # отсечение известных фантомных подписей и субтитров
        for pat in cls.HALLUCINATION_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return ""

        # удаление многократных циклических повторов фраз от 1 до 8 слов
        for n in range(8, 0, -1):
            pattern = r'(\b(?:[a-zA-Zа-яА-ЯёЁ0-9\-\,\.]+\s*){' + str(n) + r'})\s*(?:\1\s*)+'
            cleaned = re.sub(pattern, r'\1 ', cleaned, flags=re.IGNORECASE)

        # удаление зациклившихся одинаковых слов подряд
        cleaned = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\.{2,}', '...', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if re.fullmatch(r'[\s\.\,\-\:\;\!\?]+', cleaned):
            return ""
        return cleaned

    # транскрибация аудиосигнала в текст с защитой от зацикливания
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        is_music: bool = False,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        clean_path = convert_to_wav(audio_path, target_sr=16000)
        target_model = model_name or self.model_size
        model = self._load_model(target_model)
        segments_data = []
        full_text_list = []

        try:
            with silence_stderr():
                transcribe_kwargs = {
                    "beam_size": 5,
                    "best_of": 5,
                    "condition_on_previous_text": not is_music
                }

                if is_music:
                    transcribe_kwargs.update({
                        "no_speech_threshold": 0.8,
                        "compression_ratio_threshold": 2.6,
                        "temperature": (0.0, 0.2, 0.4, 0.6)
                    })
                else:
                    transcribe_kwargs.update({
                        "no_speech_threshold": 0.6,
                        "compression_ratio_threshold": 2.4,
                        "temperature": 0.0
                    })

                if language:
                    transcribe_kwargs["language"] = language
                if prompt and prompt.strip():
                    transcribe_kwargs["initial_prompt"] = prompt.strip()

                res = model.transcribe(clean_path, **transcribe_kwargs)

            raw_segs = res.get("segments", [])
            last_added_text = ""

            for idx, s in enumerate(raw_segs, 1):
                text_raw = s.get("text", "")
                text_clean = self._clean_text(text_raw)
                if not text_clean:
                    continue

                if text_clean.lower() == last_added_text.lower():
                    continue

                start_sec = round(float(s.get("start", 0)), 2)
                end_sec = round(float(s.get("end", 0)), 2)

                segments_data.append({
                    "id": idx,
                    "start": start_sec,
                    "end": end_sec,
                    "text": text_clean,
                    "start_str": self.format_timestamp(start_sec),
                    "end_str": self.format_timestamp(end_sec)
                })
                full_text_list.append(text_clean)
                last_added_text = text_clean

            detected_lang = res.get("language", "ru")
            dur = segments_data[-1]["end"] if segments_data else 0.0

            return {
                "status": "success",
                "text": " ".join(full_text_list),
                "segments": segments_data,
                "language": detected_lang,
                "duration": dur
            }
        except Exception as err:
            return {
                "status": "error",
                "text": "Не удалось извлечь речь из файла",
                "segments": [],
                "language": "ru",
                "duration": 0.0,
                "error": str(err)
            }

    # форматирование секунд в формат часы минуты и секунды
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        total_sec = int(seconds)
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        secs = total_sec % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
