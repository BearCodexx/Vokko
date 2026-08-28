import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
import torch
from backend.app.core.config import PROCESSED_DIR
from backend.app.core.audio_utils import load_and_normalize_audio
from backend.app.core.silence import silence_stderr
from backend.app.core.logger import log_info, log_error

# модуль выделения вокала нейросетью Demucs в исходном студийном качестве
class DemucsEngine:

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    # загрузка нейросетевой модели разделения
    def _load_model(self):
        if self._model is not None:
            return self._model
        import demucs.pretrained
        with silence_stderr():
            try:
                self._model = demucs.pretrained.get_model("htdemucs").to(self.device)
                log_info(f"Модель Demucs htdemucs готова на устройстве {self.device}")
                return self._model
            except Exception as err:
                log_error(f"Не удалось загрузить модель Demucs: {err}")
                return None

    # разделение дорожки на составляющие, извлечение вокала на графическом процессоре
    def isolate_vocals(self, audio_path: str, task_id: str) -> str:
        target_dir = self.output_dir / f"demucs_{task_id}"
        target_dir.mkdir(parents=True, exist_ok=True)

        vocal_output_path = target_dir / "vocals.wav"
        if vocal_output_path.exists():
            return str(vocal_output_path)

        model = self._load_model()
        if model is not None:
            try:
                from demucs.apply import apply_model
                stereo_data, sample_rate = load_and_normalize_audio(audio_path, target_sr=model.samplerate, layout="stereo")
                tensor = torch.from_numpy(stereo_data).float().to(self.device)

                with torch.no_grad():
                    sources = apply_model(model, tensor[None], device=self.device, shifts=0, split=True, overlap=0.25)[0]

                vocal_idx = model.sources.index("vocals")
                vocal_np = sources[vocal_idx].mean(dim=0).cpu().numpy()

                # нормализация уровня громкости изолированного вокала
                max_val = np.max(np.abs(vocal_np)) or 1.0
                vocal_np = (vocal_np / max_val) * 0.95

                sf.write(str(vocal_output_path), vocal_np, model.samplerate, subtype="PCM_16")
                log_info("Изоляция вокала успешно завершена нейросетью Demucs")
                return str(vocal_output_path)
            except Exception as sep_err:
                log_error(f"Сбой разделения Demucs: {sep_err}, переключение на фильтрацию")

        # резервный алгоритм очистки вокальной дорожки от низкочастотного гула
        return self._spectral_voice_filter(audio_path, str(vocal_output_path))

    # частотная фильтрация вокала, подавление баса и барабанов
    def _spectral_voice_filter(self, input_file: str, output_file: str) -> str:
        try:
            mono, samplerate = load_and_normalize_audio(input_file, target_sr=16000, layout="mono")

            # частотная фильтрация, подавление суб баса ниже семидесяти герц
            fft_data = np.fft.rfft(mono)
            freqs = np.fft.rfftfreq(len(mono), d=1.0 / samplerate)

            # срез суб баса, сохранение средних и высоких частот голоса
            bass_mask = freqs < 75.0
            fft_data[bass_mask] *= 0.15

            filtered = np.fft.irfft(fft_data, n=len(mono))
            max_val = np.max(np.abs(filtered)) or 1.0
            filtered = (filtered / max_val) * 0.95

            sf.write(output_file, filtered, samplerate, subtype="PCM_16")
            return output_file
        except Exception:
            return input_file
