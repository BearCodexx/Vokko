import os
import sys
import time
import shutil
import pathlib
from typing import List, Dict, Any, Optional
import numpy as np
import librosa
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from backend.app.core.audio_utils import load_and_normalize_audio
from backend.app.core.logger import log_info, log_error

# Monkey patch для обхода проблемы с симлинками в Windows без прав администратора
_original_symlink = pathlib.Path.symlink_to
def _safe_symlink(self, target, target_is_directory=False):
    try:
        _original_symlink(self, target, target_is_directory)
    except OSError:
        if self.exists():
            self.unlink()
        if pathlib.Path(target).is_dir():
            shutil.copytree(target, self)
        else:
            shutil.copy2(target, self)
pathlib.Path.symlink_to = _safe_symlink

# Ленивая загрузка SpeechBrain
_classifier_cache = None
def _get_speaker_classifier():
    global _classifier_cache
    if _classifier_cache is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            log_info("Загрузка нейросети для диаризации (SpeechBrain ECAPA-TDNN)...")
            _classifier_cache = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain",
                run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
            )
        except ImportError:
            log_error("SpeechBrain не установлен. Выполните: pip install speechbrain")
            return None
        except Exception as e:
            log_error(f"Сбой загрузки SpeechBrain: {e}")
            return None
    return _classifier_cache

# цветовая палитра для динамического прогресс-бара
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_MAGENTA = "\033[95m"
CLR_DIM = "\033[90m"

# отрисовка бегущей строки прогресса диаризации в одну строку
def _render_live_diarize_bar(current: int, total: int, is_final: bool = False):
    bar_len = 26
    pct = min(100.0, (current / max(1, total)) * 100.0)
    filled = int(bar_len * (pct / 100.0))
    bar = f"{CLR_GREEN}{'=' * filled}{CLR_DIM}{'-' * (bar_len - filled)}{CLR_RESET}"
    sys.stdout.write(f"\r  [{bar}] {CLR_CYAN}{pct:5.1f}%{CLR_RESET} ({current}/{total} сегментов) | {CLR_MAGENTA}Нейроанализ голосов (ИИ){CLR_RESET}   ")
    sys.stdout.flush()
    if is_final:
        sys.stdout.write("\n")
        sys.stdout.flush()

# модуль разделения спикеров, сопоставление реплик и голосов
class DiarizationEngine:

    def __init__(self, default_num_speakers: Optional[int] = None):
        self.default_num_speakers = default_num_speakers

    # извлечение многомерного нейросетевого профиля голоса (X-Vector) с помощью ECAPA-TDNN
    def _extract_segment_features(self, audio_data: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
        duration = end_sec - start_sec
        # для очень коротких реплик берем дополнительный контекст
        if duration < 0.8:
            pad = (0.8 - duration) / 2.0
            start_sec = max(0.0, start_sec - pad)
            end_sec = min(len(audio_data) / sr, end_sec + pad)

        start_sample = int(max(0, start_sec * sr))
        end_sample = int(min(len(audio_data), end_sec * sr))

        if end_sample <= start_sample + 320:
            return np.zeros(192, dtype=np.float32)

        chunk = audio_data[start_sample:end_sample]

        # Базовая очистка от пауз и фонового шума
        intervals = librosa.effects.split(chunk, top_db=30)
        if len(intervals) > 0:
            chunk = np.concatenate([chunk[s:e] for s, e in intervals])

        if len(chunk) < sr * 0.3:
            # если после очистки от тишины осталось меньше 0.3 сек звука, добиваем нулями
            chunk = np.pad(chunk, (0, int(sr * 0.3) - len(chunk)))

        classifier = _get_speaker_classifier()
        if not classifier:
            return np.zeros(192, dtype=np.float32)

        try:
            # Передаем аудиосигнал в тензор и извлекаем x-vector (размерность 192)
            tensor_chunk = torch.from_numpy(chunk).unsqueeze(0).to(classifier.device)
            # speechbrain ожидает 16кГц
            emb = classifier.encode_batch(tensor_chunk)
            return emb.squeeze().cpu().numpy()
        except Exception:
            return np.zeros(192, dtype=np.float32)

    # диаризация списка сегментов, присвоение меток спикеров
    def diarize(self, audio_path: str, segments: List[Dict[str, Any]], num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.diarize_segments(audio_path, segments, num_speakers=num_speakers)

    # диаризация списка сегментов, присвоение меток спикеров
    def diarize_segments(self, audio_path: str, segments: List[Dict[str, Any]], num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        if not segments:
            return []

        total_segments = len(segments)
        _render_live_diarize_bar(0, total_segments, is_final=False)

        try:
            mono, sr = load_and_normalize_audio(audio_path, target_sr=16000)

            feature_list = []
            last_log_time = 0.0
            for idx, seg in enumerate(segments, 1):
                feat = self._extract_segment_features(mono, sr, seg["start"], seg["end"])
                feature_list.append(feat)

                now = time.time()
                if now - last_log_time >= 0.25 or idx == total_segments or idx % 25 == 0:
                    last_log_time = now
                    _render_live_diarize_bar(idx, total_segments, is_final=(idx == total_segments))

            x = np.array(feature_list)
            
            # подстановка ближайших валидных эмбеддингов вместо нулевых векторов
            zero_masks = np.all(x == 0, axis=1)
            if np.any(zero_masks) and not np.all(zero_masks):
                valid_indices = np.where(~zero_masks)[0]
                for z_idx in np.where(zero_masks)[0]:
                    nearest = valid_indices[np.argmin(np.abs(valid_indices - z_idx))]
                    x[z_idx] = x[nearest]

            # Нормализация на единичную сферу (L2-норма)
            x_norm = normalize(x)

            n_samples = len(segments)
            if num_speakers is None:
                n_clusters = 2 if n_samples >= 4 else 1
            else:
                n_clusters = max(1, min(num_speakers, n_samples))

            if n_clusters > 1 and n_samples >= n_clusters:
                # Ward-linkage на L2-нормализованных векторах предотвращает коллапс в один кластер из-за выбросов
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
                raw_labels = clustering.fit_predict(x_norm)
            else:
                raw_labels = [0] * n_samples

            # Хронологическая нумерация спикеров по порядку их появления в аудио
            speaker_mapping = {}
            next_speaker_id = 1
            final_labels = []
            for l in raw_labels:
                if l not in speaker_mapping:
                    speaker_mapping[l] = next_speaker_id
                    next_speaker_id += 1
                final_labels.append(speaker_mapping[l])

            result_segments = []
            for idx, seg in enumerate(segments):
                label_id = int(final_labels[idx])
                updated = dict(seg)
                updated["speaker"] = f"Спикер {label_id}"
                updated["speaker_id"] = label_id
                result_segments.append(updated)

            log_info(f"Диаризация завершена: 100% | Выделено спикеров: {len(speaker_mapping)}")
            return result_segments
        except Exception as e:
            log_error(f"Сбой диаризации: {e}")
            for s in segments:
                s["speaker"] = "Спикер 1"
                s["speaker_id"] = 1
            return segments

    # переименование спикера по всему тексту, замена метки
    @staticmethod
    def rename_speaker(segments: List[Dict[str, Any]], old_name: str, new_name: str) -> List[Dict[str, Any]]:
        for seg in segments:
            if seg.get("speaker") == old_name:
                seg["speaker"] = new_name
        return segments

    # удаление реплик указанного спикера, очистка списка
    @staticmethod
    def delete_speaker(segments: List[Dict[str, Any]], speaker_name: str) -> List[Dict[str, Any]]:
        return [seg for seg in segments if seg.get("speaker") != speaker_name]

    # слияние двух спикеров, перенос реплик
    @staticmethod
    def merge_speakers(segments: List[Dict[str, Any]], source_speaker: str, target_speaker: str) -> List[Dict[str, Any]]:
        for seg in segments:
            if seg.get("speaker") == source_speaker:
                seg["speaker"] = target_speaker
        return segments
