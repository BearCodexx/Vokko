import os
import re
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import yt_dlp
import backend.app.core.doh
from backend.app.core.config import UPLOAD_DIR
from backend.app.core.audio_utils import convert_to_wav
from backend.app.core.logger import log_info, log_error

# модуль загрузки звуковых дорожек с открытых медиахостингов
class MediaDownloader:

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or UPLOAD_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # подготовка параметров загрузки, обход проверок ботов
    def _build_ydl_options(self, target_id: str) -> Dict[str, Any]:
        output_template = str(self.output_dir / f"{target_id}.%(ext)s")
        return {
            "format": "ba/b[ext=m4a]/b[ext=mp4]/b",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ignoreerrors": False,
            "nocheckcertificate": True,
            "socket_timeout": 60,
            "retries": 10,
            "fragment_retries": 10,
            "extractor_retries": 5,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web_creator", "ios"]
                }
            }
        }

    # скачивание дорожки по ссылке с открытых платформ
    def download_url(self, url: str) -> Dict[str, Any]:
        clean_url = url.strip()

        # отклонение ссылок с платформ с платными ограничениями
        if "music.yandex" in clean_url or "yandex.ru/music" in clean_url:
            raise RuntimeError("Яндекс Музыка блокирует доступ к полным песням без платной подписки, загрузите аудиофайл песни напрямую в окно загрузки или укажите ссылку на VK Видео, RuTube, YouTube или Soundcloud")

        task_id = str(uuid.uuid4())[:8]
        ydl_opts = self._build_ydl_options(task_id)
        title = "Аудиозапись"
        duration = 0.0
        artist = ""
        raw_path = None

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=True)
                if info:
                    title = info.get("title", title)
                    duration = float(info.get("duration", 0.0) or 0.0)
                    artist = info.get("artist", "") or info.get("uploader", "")
                    try:
                        prep = ydl.prepare_filename(info)
                        if os.path.exists(prep):
                            raw_path = prep
                    except Exception:
                        pass
        except Exception as dl_err:
            log_error(f"Предупреждение загрузчика: {dl_err}")
            raw_path = self._try_direct_or_stream_download(clean_url, task_id)

        if not raw_path or not os.path.exists(raw_path):
            for candidate in self.output_dir.glob(f"{task_id}.*"):
                if candidate.is_file() and not candidate.name.endswith((".wav", ".part")):
                    raw_path = str(candidate)
                    break

        if not raw_path or not os.path.exists(raw_path):
            raise RuntimeError(f"Не удалось загрузить аудио по адресу: {clean_url}")

        clean_wav = self.output_dir / f"{task_id}_clean.wav"
        final_path = convert_to_wav(raw_path, str(clean_wav), target_sr=16000)

        return {
            "file_path": final_path,
            "title": title,
            "duration": duration,
            "artist": artist,
            "task_id": task_id
        }

    # запасной вариант загрузки прямого звукового потока
    def _try_direct_or_stream_download(self, url: str, task_id: str) -> Optional[str]:
        target = self.output_dir / f"{task_id}.audio"
        try:
            import requests
            resp = requests.get(url, stream=True, timeout=30, verify=False)
            content_type = resp.headers.get("Content-Type", "").lower()
            # отклонение веб страниц и текстовых ответов
            if "text/html" in content_type or "text/plain" in content_type:
                return None
            if resp.status_code == 200:
                with open(target, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                if target.stat().st_size > 1024 * 64:
                    return str(target)
        except Exception:
            return None
        return None
