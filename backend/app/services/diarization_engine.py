import os
import sys
import time
import shutil
import pathlib
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from backend.app.core.audio_utils import load_and_normalize_audio
from backend.app.core.logger import log_info, log_error

# обход проблемы с символическими ссылками в среде windows
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

# отложенная загрузка нейросети распознавания голоса
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
            log_error("SpeechBrain не установлен, выполните pip install speechbrain")
            return None
        except Exception as e:
            log_error(f"Сбой загрузки SpeechBrain: {e}")
            return None
    return _classifier_cache

# цветовая палитра для динамического прогресс бара
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

    # извлечение многомерного нейросетевого профиля голоса
    def _extract_segment_features(self, audio_data: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
        start_sample = int(max(0, start_sec * sr))
        end_sample = int(min(len(audio_data), end_sec * sr))

        if end_sample <= start_sample + 320:
            return np.zeros(192, dtype=np.float32)

        chunk = audio_data[start_sample:end_sample].copy()

        # для коротких реплик применяем мягкое зеркальное дополнение до минимального окна фильтров
        min_samples = int(sr * 0.75)
        if len(chunk) < min_samples:
            if len(chunk) > 0:
                pad_len = min_samples - len(chunk)
                chunk = np.pad(chunk, (0, pad_len), mode="reflect")
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
            # для длинных фрагментов вычисляем усредненный профиль по скользящему окну
            dur_sec = len(chunk) / sr
            if dur_sec > 3.5:
                win_samples = int(2.0 * sr)
                hop_samples = int(1.0 * sr)
                sub_embs = []
                for pos in range(0, len(chunk) - win_samples + 1, hop_samples):
                    sub_chunk = chunk[pos : pos + win_samples]
                    tensor_sub = torch.from_numpy(sub_chunk).unsqueeze(0).to(classifier.device)
                    with torch.no_grad():
                        sub_feat = classifier.encode_batch(tensor_sub).squeeze().cpu().numpy()
                    sub_norm = np.linalg.norm(sub_feat)
                    if sub_norm > 1e-6:
                        sub_embs.append(sub_feat / sub_norm)
                if sub_embs:
                    avg_feat = np.mean(sub_embs, axis=0)
                    norm_val = np.linalg.norm(avg_feat)
                    if norm_val > 1e-6:
                        return (avg_feat / norm_val).astype(np.float32)

            tensor_chunk = torch.from_numpy(chunk).unsqueeze(0).to(classifier.device)
            with torch.no_grad():
                emb = classifier.encode_batch(tensor_chunk).squeeze().cpu().numpy()
            norm_val = np.linalg.norm(emb)
            if norm_val > 1e-6:
                return (emb / norm_val).astype(np.float32)
            return emb.astype(np.float32)
        except Exception:
            return np.zeros(192, dtype=np.float32)

    # иерархическая косинусная кластеризация спикеров с автоопределением числа голосов
    def _cluster_speaker_embeddings(self, x: np.ndarray, num_speakers: Optional[int] = None, max_candidates: int = 8) -> np.ndarray:
        n_samples = len(x)
        if n_samples < 2:
            return np.zeros(n_samples, dtype=int)

        # нормализация признаков на единичную сферу
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        x_norm = x / norms

        # матрица косинусных расстояний
        dist_mat = cosine_distances(x_norm)
        np.fill_diagonal(dist_mat, 0.0)

        # ручной выбор точного числа спикеров пользователем
        if num_speakers is not None and num_speakers >= 1:
            k = min(int(num_speakers), n_samples)
            if k == 1:
                return np.zeros(n_samples, dtype=int)
            clustering = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            return clustering.fit_predict(dist_mat)

        # проверка на одноголосную речь, монолог
        triu_dists = dist_mat[np.triu_indices(n_samples, k=1)]
        if len(triu_dists) > 0:
            if np.percentile(triu_dists, 85) < 0.35 or np.mean(triu_dists) < 0.28:
                return np.zeros(n_samples, dtype=int)

        # автоматический подбор оптимального числа спикеров
        max_k = min(max_candidates, max(2, n_samples // 3))
        best_k = 2
        best_score = -999.0

        for k in range(2, max_k + 1):
            clustering = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            cand_labels = clustering.fit_predict(dist_mat)

            counts = np.bincount(cand_labels)
            singletons = np.sum(counts <= 1)
            if singletons > max(1, k // 2):
                continue

            try:
                sil = silhouette_score(dist_mat, cand_labels, metric="precomputed")
            except Exception:
                sil = -1.0

            # расчет косинусного расстояния между центроидами спикеров
            centroids = []
            for l in range(k):
                mask = (cand_labels == l)
                if np.sum(mask) > 0:
                    c = np.mean(x_norm[mask], axis=0)
                    c_norm = np.linalg.norm(c)
                    centroids.append(c / max(1e-6, c_norm))

            min_cent_dist = 1.0
            if len(centroids) > 1:
                cent_dists = cosine_distances(np.array(centroids))
                np.fill_diagonal(cent_dists, 1.0)
                min_cent_dist = float(np.min(cent_dists))

            # штраф за слишком близкие центроиды, принадлежащие одному человеку
            if min_cent_dist < 0.15:
                continue

            score = sil + 0.35 * min_cent_dist
            if score > best_score:
                best_score = score
                best_k = k

        final_clustering = AgglomerativeClustering(n_clusters=best_k, metric="precomputed", linkage="average")
        raw_labels = final_clustering.fit_predict(dist_mat)

        # перепривязка редких одиночных шумов к ближайшим установленным центроидам
        counts = np.bincount(raw_labels)
        main_clusters = [l for l in range(best_k) if counts[l] >= 2]
        if main_clusters and len(main_clusters) < best_k:
            cent_map = {}
            for l in main_clusters:
                c = np.mean(x_norm[raw_labels == l], axis=0)
                cent_map[l] = c / max(1e-6, np.linalg.norm(c))
            for idx in range(n_samples):
                if counts[raw_labels[idx]] < 2:
                    best_main = min(main_clusters, key=lambda l: 1.0 - float(np.dot(x_norm[idx], cent_map[l])))
                    if (1.0 - float(np.dot(x_norm[idx], cent_map[best_main]))) < 0.70:
                        raw_labels[idx] = best_main

        # переиндексация меток кластеров по порядку
        unique_labels = sorted(list(set(raw_labels)))
        remap = {old: new for new, old in enumerate(unique_labels)}
        return np.array([remap[l] for l in raw_labels], dtype=int)

    # темпоральное сглаживание, устраняет случайные одиночные перескоки спикера внутри непрерывной речи
    def _smooth_speaker_labels(self, labels: List[int], segments: List[Dict[str, Any]], max_gap_sec: float = 0.8) -> List[int]:
        smoothed = list(labels)
        n = len(labels)
        for i in range(1, n - 1):
            prev_spk, curr_spk, next_spk = smoothed[i - 1], smoothed[i], smoothed[i + 1]
            if prev_spk == next_spk and curr_spk != prev_spk:
                gap1 = float(segments[i]["start"]) - float(segments[i - 1]["end"])
                gap2 = float(segments[i + 1]["start"]) - float(segments[i]["end"])
                dur = float(segments[i]["end"]) - float(segments[i]["start"])
                if (gap1 < max_gap_sec or gap2 < max_gap_sec) and dur < 2.5:
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
            # гарантируем загрузку модели до начала отрисовки бегущего прогресс бара
            _get_speaker_classifier()
            mono, sr = load_and_normalize_audio(audio_path, target_sr=16000)

            _render_live_diarize_bar(0, total_segments, is_final=False)

            feature_list = []
            last_log_time = 0.0
            for idx, seg in enumerate(segments, 1):
                feat = self._extract_segment_features(mono, sr, float(seg["start"]), float(seg["end"]))
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

            # выполнение косинусной кластеризации
            target_speakers = num_speakers if num_speakers is not None else self.default_num_speakers
            raw_labels = self._cluster_speaker_embeddings(x, num_speakers=target_speakers)

            # сглаживание случайных микроперескоков спикера
            smoothed_labels = self._smooth_speaker_labels(raw_labels, segments)

            # упорядочивание спикеров по суммарной длительности речи
            cluster_durs = {}
            for idx, lbl in enumerate(smoothed_labels):
                dur = float(segments[idx].get("end", 0.0)) - float(segments[idx].get("start", 0.0))
                cluster_durs[lbl] = cluster_durs.get(lbl, 0.0) + max(0.2, dur)

            sorted_clusters = sorted(cluster_durs.keys(), key=lambda c: cluster_durs[c], reverse=True)
            speaker_mapping = {c: idx + 1 for idx, c in enumerate(sorted_clusters)}

            result_segments = []
            for idx, seg in enumerate(segments):
                raw_lbl = smoothed_labels[idx]
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
