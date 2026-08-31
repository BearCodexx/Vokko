import os
import sys
import re
import time
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.app.core.audio_utils import convert_to_wav, get_audio_duration
from backend.app.core.silence import silence_stderr
from backend.app.core.logger import log_info, log_error
from backend.app.core.progress import set_task_progress
from backend.app.core.models_manager import (
    WHISPER_MODELS,
    get_model_path,
    download_model_file,
    is_model_downloaded,
    load_app_config
)

# цветовая палитра для динамического прогресс-бара
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_MAGENTA = "\033[95m"
CLR_DIM = "\033[90m"

# отрисовка бегущей строки прогресса транскрибации в одну строку
def _render_live_transcribe_bar(current_sec: float, total_sec: float, segments_count: int, is_final: bool = False):
    bar_len = 26
    pct = min(100.0, (current_sec / max(1.0, total_sec)) * 100.0) if total_sec > 0 else 0.0
    filled = int(bar_len * (pct / 100.0))
    bar = f"{CLR_GREEN}{'=' * filled}{CLR_DIM}{'-' * (bar_len - filled)}{CLR_RESET}"
    time_curr = WhisperEngine.format_timestamp(current_sec)
    time_total = WhisperEngine.format_timestamp(total_sec) if total_sec > 0 else "..."
    sys.stdout.write(f"\r  [{bar}] {CLR_CYAN}{pct:5.1f}%{CLR_RESET} ({time_curr} / {time_total}) | {CLR_MAGENTA}Сегментов: {segments_count}{CLR_RESET}   ")
    sys.stdout.flush()
    if is_final:
        sys.stdout.write("\n")
        sys.stdout.flush()

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
        self._faster_model = None
        self._faster_loaded_name = None
        self._openai_model = None
        self._openai_loaded_name = None

    # загрузка оптимизированной модели faster whisper на ctranslate2
    def _load_faster_model(self, target_name: Optional[str] = None):
        model_name = target_name or self.model_size
        if self._faster_model is not None and self._faster_loaded_name == model_name:
            return self._faster_model

        from faster_whisper import WhisperModel
        from backend.app.core.config import MODELS_WHISPER_DIR
        compute_type = "float16" if self.device == "cuda" else "int8"

        # проверка локально скачанной модели
        local_dir = MODELS_WHISPER_DIR / "faster" / model_name
        if local_dir.exists() and any(f.name.endswith((".bin", ".safetensors")) for f in local_dir.glob("*")):
            model_id = str(local_dir)
            log_info(f"Загрузка локальной модели faster-whisper из {local_dir} на устройстве {self.device}")
        else:
            # приведение внутренних имен моделей к формату faster whisper
            model_id = model_name
            if model_name == "turbo":
                model_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
            elif model_name == "large-v3":
                model_id = "large-v3"
            log_info(f"Загрузка движка faster-whisper ({model_id}) на устройстве {self.device} (при первом использовании выполняется загрузка весов)...")

        try:
            self._faster_model = WhisperModel(model_id, device=self.device, compute_type=compute_type)
            self._faster_loaded_name = model_name
            log_info(f"Модель faster-whisper ({model_name}) успешно загружена в память GPU")
            return self._faster_model
        except Exception as e:
            log_error(f"Не удалось загрузить faster-whisper на GPU, пробуем CPU: {e}")
            self._faster_model = WhisperModel(model_id, device="cpu", compute_type="int8")
            self._faster_loaded_name = model_name
            return self._faster_model

    # загрузка стандартной модели openai whisper на pytorch
    def _load_openai_model(self, target_name: Optional[str] = None):
        model_name = target_name or self.model_size
        if self._openai_model is not None and self._openai_loaded_name == model_name:
            return self._openai_model

        import whisper
        if not is_model_downloaded(model_name, engine="openai-whisper"):
            log_info(f"Скачивание весов модели Whisper {model_name}")
            download_model_file(model_name, engine="openai-whisper")

        model_path = get_model_path(model_name, engine="openai-whisper")
        load_target = str(model_path) if model_path and model_path.exists() else model_name

        with silence_stderr():
            try:
                self._openai_model = whisper.load_model(load_target, device=self.device)
                self._openai_loaded_name = model_name
                log_info(f"Модель openai-whisper {model_name} готова на устройстве {self.device}")
                return self._openai_model
            except Exception as e:
                log_error(f"Не удалось загрузить openai-whisper на GPU: {e}")
                self._openai_model = whisper.load_model("base", device="cpu")
                self._openai_loaded_name = "base"
                return self._openai_model

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

    # транскрибация аудиосигнала через быстрый движок faster whisper c vad фильтрацией
    def _transcribe_faster(
        self,
        clean_path: str,
        target_model: str,
        beam_size: int,
        language: Optional[str],
        prompt: Optional[str],
        is_music: bool,
        task_id: Optional[str],
        total_duration: float
    ) -> Dict[str, Any]:
        model = self._load_faster_model(target_model)
        segments_data = []
        full_text_list = []
        last_log_time = 0.0

        # запуск генератора распознавания с фильтрацией пауз
        log_info(f"Старт декодирования аудиопотока через faster-whisper (длительность: {self.format_timestamp(total_duration)})...")
        segments_gen, info = model.transcribe(
            clean_path,
            beam_size=beam_size,
            vad_filter=not is_music,
            initial_prompt=prompt if prompt else None,
            language=language if language else None,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if not is_music else (0.0, 0.2, 0.4),
            condition_on_previous_text=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            word_timestamps=True
        )

        detected_lang = info.language or "ru"
        last_added_text = ""
        seg_idx = 1

        for raw_s in segments_gen:
            words = getattr(raw_s, "words", None)
            # если есть пословные таймкоды, нарезаем реплики по естественным паузам (>0.45с) для чистой диаризации
            sub_segments = []
            if words and len(words) > 0:
                cur_words = []
                cur_start = None
                for w_idx, w in enumerate(words):
                    if cur_start is None:
                        cur_start = float(w.start)
                    cur_words.append(w.word)
                    is_last = (w_idx == len(words) - 1)
                    has_pause = False
                    if not is_last:
                        next_w = words[w_idx + 1]
                        gap = float(next_w.start) - float(w.end)
                        cur_dur = float(w.end) - cur_start
                        # разделяем реплики только при настоящей паузе
                        if gap >= 0.55 or (cur_dur >= 7.5 and gap >= 0.35):
                            has_pause = True
                    if (is_last or has_pause) and cur_words:
                        sub_text = self._clean_text(" ".join(cur_words))
                        if sub_text:
                            sub_segments.append((round(cur_start, 2), round(float(w.end), 2), sub_text))
                        cur_words = []
                        cur_start = None
            else:
                raw_text = self._clean_text(raw_s.text or "")
                if raw_text:
                    sub_segments.append((round(float(raw_s.start), 2), round(float(raw_s.end), 2), raw_text))

            for start_sec, end_sec, text_clean in sub_segments:
                if text_clean.lower() == last_added_text.lower():
                    continue

                segments_data.append({
                    "id": seg_idx,
                    "start": start_sec,
                    "end": end_sec,
                    "text": text_clean,
                    "start_str": self.format_timestamp(start_sec),
                    "end_str": self.format_timestamp(end_sec)
                })
                full_text_list.append(text_clean)
                last_added_text = text_clean
                seg_idx += 1

            # динамический вывод прогресса в одну строку консоли и обновление статуса
            raw_end = round(float(raw_s.end), 2)
            now = time.time()
            if now - last_log_time >= 0.5 or seg_idx % 5 == 0:
                last_log_time = now
                _render_live_transcribe_bar(raw_end, total_duration, len(segments_data), is_final=False)
                if task_id:
                    pct = min(99, int((raw_end / max(1.0, total_duration)) * 100)) if total_duration > 0 else 0
                    time_curr = self.format_timestamp(round(float(raw_s.start), 2))
                    time_total = self.format_timestamp(total_duration) if total_duration > 0 else "..."
                    set_task_progress(task_id, 2, 3, f"Распознавание речи Whisper: {pct}% ({time_curr} / {time_total})")

        _render_live_transcribe_bar(total_duration, total_duration, len(segments_data), is_final=True)
        dur = segments_data[-1]["end"] if segments_data else total_duration
        log_info(f"Транскрибация завершена: 100% | Всего распознано сегментов: {len(segments_data)}")

        return {
            "status": "success",
            "text": " ".join(full_text_list),
            "segments": segments_data,
            "language": detected_lang,
            "duration": dur
        }

    # транскрибация аудиосигнала через классический движок openai whisper
    def _transcribe_openai(
        self,
        clean_path: str,
        target_model: str,
        beam_size: int,
        language: Optional[str],
        prompt: Optional[str],
        is_music: bool
    ) -> Dict[str, Any]:
        model = self._load_openai_model(target_model)
        segments_data = []
        full_text_list = []

        with silence_stderr():
            transcribe_kwargs = {
                "beam_size": beam_size,
                "best_of": beam_size,
                "condition_on_previous_text": False,
                "word_timestamps": True
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
                    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                })

            if language:
                transcribe_kwargs["language"] = language
            if prompt and prompt.strip():
                transcribe_kwargs["initial_prompt"] = prompt.strip()

            res = model.transcribe(clean_path, **transcribe_kwargs)

        raw_segs = res.get("segments", [])
        last_added_text = ""
        seg_idx = 1

        for raw_s in raw_segs:
            words = raw_s.get("words", [])
            sub_segments = []
            if words and len(words) > 0:
                cur_words = []
                cur_start = None
                for w_idx, w in enumerate(words):
                    w_start = float(w.get("start", 0.0))
                    w_end = float(w.get("end", 0.0))
                    w_word = str(w.get("word", "")).strip()
                    if cur_start is None:
                        cur_start = w_start
                    cur_words.append(w_word)
                    is_last = (w_idx == len(words) - 1)
                    has_pause = False
                    if not is_last:
                        next_w = words[w_idx + 1]
                        next_start = float(next_w.get("start", 0.0))
                        gap = next_start - w_end
                        cur_dur = w_end - cur_start
                        if gap >= 0.55 or (cur_dur >= 7.5 and gap >= 0.35):
                            has_pause = True
                    if (is_last or has_pause) and cur_words:
                        sub_text = self._clean_text(" ".join(cur_words))
                        if sub_text:
                            sub_segments.append((round(cur_start, 2), round(w_end, 2), sub_text))
                        cur_words = []
                        cur_start = None
            else:
                raw_text = self._clean_text(raw_s.get("text", ""))
                if raw_text:
                    sub_segments.append((round(float(raw_s.get("start", 0)), 2), round(float(raw_s.get("end", 0)), 2), raw_text))

            for start_sec, end_sec, text_clean in sub_segments:
                if text_clean.lower() == last_added_text.lower():
                    continue

                segments_data.append({
                    "id": seg_idx,
                    "start": start_sec,
                    "end": end_sec,
                    "text": text_clean,
                    "start_str": self.format_timestamp(start_sec),
                    "end_str": self.format_timestamp(end_sec)
                })
                full_text_list.append(text_clean)
                last_added_text = text_clean
                seg_idx += 1

        detected_lang = res.get("language", "ru")
        dur = segments_data[-1]["end"] if segments_data else 0.0

        return {
            "status": "success",
            "text": " ".join(full_text_list),
            "segments": segments_data,
            "language": detected_lang,
            "duration": dur
        }

    # единая точка входа для транскрибации речи и музыки
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        is_music: bool = False,
        model_name: Optional[str] = None,
        engine_type: str = "faster-whisper",
        beam_size: int = 1,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        clean_path = convert_to_wav(audio_path, target_sr=16000)
        target_model = model_name or self.model_size
        total_duration = get_audio_duration(clean_path)

        # ограничение параметра поиска по лучам в безопасных границах
        clean_beam = max(1, min(5, int(beam_size)))

        try:
            if engine_type == "openai-whisper":
                log_info(f"Запуск распознавания openai-whisper ({target_model}, beam: {clean_beam})")
                return self._transcribe_openai(
                    clean_path=clean_path,
                    target_model=target_model,
                    beam_size=clean_beam,
                    language=language,
                    prompt=prompt,
                    is_music=is_music
                )
            else:
                log_info(f"Запуск распознавания faster-whisper ({target_model}, beam: {clean_beam})")
                return self._transcribe_faster(
                    clean_path=clean_path,
                    target_model=target_model,
                    beam_size=clean_beam,
                    language=language,
                    prompt=prompt,
                    is_music=is_music,
                    task_id=task_id,
                    total_duration=total_duration
                )
        except Exception as err:
            log_error(f"Сбой при выполнении транскрибации: {err}")
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

