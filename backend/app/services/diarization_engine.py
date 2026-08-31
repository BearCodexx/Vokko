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
from sklearn.metrics import silhouette_score
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
        start_sample = int(max(0, start_sec * sr))
        end_sample = int(min(len(audio_data), end_sec * sr))

        if end_sample <= start_sample + 320:
            return np.zeros(192, dtype=np.float32)

        chunk = audio_data[start_sample:end_sample].copy()

        # для коротких реплик циклически дублируем сам фрагмент, никогда не залезая в чужие соседние реплики
        if len(chunk) < int(sr * 0.9):
            if len(chunk) > 0:
                rep = int(np.ceil((sr * 0.9) / len(chunk)))
                chunk = np.tile(chunk, rep)[:int(sr * 0.9)]
            else:
                return np.zeros(192, dtype=np.float32)

        # нормализация амплитуды без искусственной нарезки, сохраняющая естественный тембр
        max_val = np.max(np.abs(chunk))
        if max_val > 1e-4:
            chunk = chunk / max_val

        classifier = _get_speaker_classifier()
        if not classifier:
            return np.zeros(192, dtype=np.float32)

        try:
            tensor_chunk = torch.from_numpy(chunk).unsqueeze(0).to(classifier.device)
            emb = classifier.encode_batch(tensor_chunk)
            return emb.squeeze().cpu().numpy()
        except Exception:
            return np.zeros(192, dtype=np.float32)

    # автоматическое определение оптимального числа спикеров через расстояние между центроидами
    def _auto_detect_speakers_count(self, x_norm: np.ndarray, max_candidates: int = 6) -> int:
        n_samples = len(x_norm)
        if n_samples < 4:
            return 1

        max_k = min(max_candidates, max(2, n_samples // 4))
        best_k = 1
        best_score = -1.0

        for k in range(2, max_k + 1):
            clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = clustering.fit_predict(x_norm)

            counts = np.bincount(labels)
            if np.min(counts) < max(2, int(n_samples * 0.03)):
                continue

            centroids = [normalize(np.mean(x_norm[labels == l], axis=0, keepdims=True)).squeeze() for l in range(k)]
            min_dist = min([1.0 - float(np.dot(centroids[i], centroids[j])) for i in range(k) for j in range(i + 1, k)])

            # если центроиды двух групп ближе 0.28 косинусного расстояния — они принадлежат одному человеку
            if min_dist < 0.28:
                continue

            try:
                sil = silhouette_score(x_norm, labels)
                if sil > best_score and sil > 0.07:
                    best_score = sil
                    best_k = k
            except Exception:
                continue

        return best_k

    # темпоральное сглаживание: устраняет случайные одиночные перескоки спикера внутри непрерывной речи
    def _smooth_speaker_labels(self, labels: List[int], segments: List[Dict[str, Any]], max_gap_sec: float = 0.6) -> List[int]:
        smoothed = list(labels)
        n = len(labels)
        for i in range(1, n - 1):
            prev_spk, curr_spk, next_spk = smoothed[i - 1], smoothed[i], smoothed[i + 1]
            if prev_spk == next_spk and curr_spk != prev_spk:
                gap1 = segments[i]["start"] - segments[i - 1]["end"]
                gap2 = segments[i + 1]["start"] - segments[i]["end"]
                dur = segments[i]["end"] - segments[i]["start"]
                if (gap1 < max_gap_sec or gap2 < max_gap_sec) and dur < 3.5:
                    smoothed[i] = prev_spk
        return smoothed

    # диаризация списка сегментов, присвоение меток спикеров
    def diarize(self, audio_path: str, segments: List[Dict[str, Any]], num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.diarize_segments(audio_path, segments, num_speakers=num_speakers)

    # диаризация списка сегментов, присвоение меток спикеров
    def diarize_segments(self, audio_path: str, segments: List[Dict[str, Any]], num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        if not segments:
            return []

        total_segments = len(segments)

        try:
            # гарантируем загрузку модели до начала отрисовки бегущего прогресс-бара
            _get_speaker_classifier()
            mono, sr = load_and_normalize_audio(audio_path, target_sr=16000)

            _render_live_diarize_bar(0, total_segments, is_final=False)

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

            # нормализация на единичную сферу
            x_norm = normalize(x)
            n_samples = len(segments)

            # определение количества спикеров
            if num_speakers is not None and num_speakers >= 1:
                k = max(1, min(int(num_speakers), n_samples))
            else:
                k = self._auto_detect_speakers_count(x_norm, max_candidates=6)

            if k <= 1 or n_samples < 2:
                raw_labels = [0] * n_samples
            else:
                clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
                raw_labels = clustering.fit_predict(x_norm)

            # упорядочивание спикеров по суммарной длительности речи
            cluster_durs = {}
            for idx, lbl in enumerate(raw_labels):
                dur = float(segments[idx].get("end", 0.0)) - float(segments[idx].get("start", 0.0))
                cluster_durs[lbl] = cluster_durs.get(lbl, 0.0) + max(0.2, dur)

            sorted_clusters = sorted(cluster_durs.keys(), key=lambda c: cluster_durs[c], reverse=True)
            speaker_mapping = {c: idx + 1 for idx, c in enumerate(sorted_clusters)}

            result_segments = []
            for idx, seg in enumerate(segments):
                raw_lbl = raw_labels[idx]
                label_id = speaker_mapping.get(raw_lbl, 1)
                updated = dict(seg)
                updated["speaker"] = f"Спикер {label_id}"
                updated["speaker_id"] = label_id
                result_segments.append(updated)

            log_info(f"Диаризация завершена: 100% | Выделено спикеров: {len(set(speaker_mapping.values()))}")
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
