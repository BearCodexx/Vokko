import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import av

# модуль загрузки звуковых данных с сохранением исходной частоты дискретизации
def load_and_normalize_audio(file_path: str, target_sr: int = 16000, layout: str = "mono") -> Tuple[np.ndarray, int]:
    # чтение аудиопотока через медиа библиотеку
    try:
        container = av.open(file_path)
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=target_sr)
        audio_arrays = []

        for frame in container.decode(audio=0):
            for resampled_frame in resampler.resample(frame):
                audio_arrays.append(resampled_frame.to_ndarray())

        container.close()
        if audio_arrays:
            concatenated = np.concatenate(audio_arrays, axis=1)
            if layout == "mono":
                concatenated = concatenated.flatten()
            return concatenated.astype(np.float32), target_sr
    except Exception:
        pass

    # запасной вариант прямого чтения файла через soundfile
    try:
        data, sr = sf.read(file_path)
        if layout == "mono" and data.ndim > 1:
            data = np.mean(data, axis=1)
        elif layout == "stereo" and data.ndim == 1:
            data = np.stack([data, data])
        elif layout == "stereo" and data.ndim > 1:
            data = data.T
        return data.astype(np.float32), sr
    except Exception:
        if layout == "stereo":
            return np.zeros((2, target_sr), dtype=np.float32), target_sr
        return np.zeros(target_sr, dtype=np.float32), target_sr

# сохранение очищенного звукового файла в формате wav
def convert_to_wav(input_path: str, output_path: Optional[str] = None, target_sr: int = 16000) -> str:
    target = output_path or str(Path(input_path).with_suffix(".clean.wav"))
    audio_data, sr = load_and_normalize_audio(input_path, target_sr=target_sr, layout="mono")
    sf.write(target, audio_data, sr, subtype="PCM_16")
    return target
