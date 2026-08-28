import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from backend.app.core.audio_utils import load_and_normalize_audio

# модуль разделения спикеров, сопоставление реплик и голосов
class DiarizationEngine:

    def __init__(self, default_num_speakers: Optional[int] = None):
        self.default_num_speakers = default_num_speakers

    # извлечение акустических признаков сегмента, вычисление спектра
    def _extract_segment_features(self, audio_data: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
        start_sample = int(max(0, start_sec * sr))
        end_sample = int(min(len(audio_data), end_sec * sr))

        if end_sample <= start_sample + 512:
            return np.zeros(24)

        chunk = audio_data[start_sample:end_sample]

        # вычисление базовых характеристик спектра, энергия и переходы через ноль
        fft_vals = np.abs(np.fft.rfft(chunk, n=1024))
        freqs = np.fft.rfftfreq(1024, d=1.0 / sr)

        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-9)
        spectral_energy = np.mean(chunk ** 2)
        zero_crossings = np.mean(np.abs(np.diff(np.sign(chunk))))

        # полосовые спектральные интервалы для разделения тембра
        num_bands = 21
        band_splits = np.array_split(fft_vals[:512], num_bands)
        band_energies = [float(np.mean(b)) for b in band_splits]

        features = [spectral_centroid, spectral_energy, zero_crossings] + band_energies
        return np.array(features, dtype=np.float32)

    # диаризация списка сегментов, присвоение меток спикеров
    def diarize_segments(self, audio_path: str, segments: List[Dict[str, Any]], num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        if not segments:
            return []

        try:
            mono, sr = load_and_normalize_audio(audio_path, target_sr=16000)

            feature_list = []
            for seg in segments:
                feat = self._extract_segment_features(mono, sr, seg["start"], seg["end"])
                feature_list.append(feat)

            x = np.array(feature_list)
            # нормализация матрицы признаков, центрирование
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0) + 1e-6
            x_norm = (x - mean) / std

            n_samples = len(segments)
            if num_speakers is None:
                # оценка количества говорящих по объему сегментов
                n_clusters = 2 if n_samples >= 4 else 1
            else:
                n_clusters = max(1, min(num_speakers, n_samples))

            if n_clusters > 1 and n_samples >= n_clusters:
                clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
                labels = clustering.fit_predict(x_norm)
            else:
                labels = [0] * n_samples

            result_segments = []
            for idx, seg in enumerate(segments):
                label_id = int(labels[idx]) + 1
                updated = dict(seg)
                updated["speaker"] = f"Спикер {label_id}"
                updated["speaker_id"] = label_id
                result_segments.append(updated)

            return result_segments
        except Exception:
            # в случае ошибки присваиваем единого спикера
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
