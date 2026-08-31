import os
import re
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import yt_dlp
import backend.app.core.doh
from backend.app.core.config import BASE_DIR, DATA_DIR, UPLOAD_DIR
from backend.app.core.audio_utils import convert_to_wav
from backend.app.core.logger import log_info, log_error

# модуль загрузки звуковых дорожек с открытых медиахостингов
class MediaDownloader:

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or UPLOAD_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # поиск файла cookies.txt для доступа к закрытым или 18+ видео
    def _find_cookie_file(self) -> Optional[str]:
        candidates = [
            BASE_DIR / "cookies.txt",
            DATA_DIR / "cookies.txt",
            Path("cookies.txt").resolve(),
            Path("data/cookies.txt").resolve(),
            Path("temp_storage/cookies.txt").resolve()
        ]
        for c in candidates:
            if c.exists() and c.is_file() and c.stat().st_size > 0:
                # защитная копия во временный каталог, чтобы yt-dlp не стирал токены авторизации из оригинала
                temp_cookie = self.output_dir / "session_cookies.txt"
                try:
                    shutil.copy2(c, temp_cookie)
                    log_info(f"Загружены куки авторизации из {c.name}")
                    return str(temp_cookie)
                except Exception:
                    return str(c.resolve())
        return None

    # подготовка параметров загрузки, обход проверок ботов
    def _build_ydl_options(self, target_id: str) -> Dict[str, Any]:
        output_template = str(self.output_dir / f"{target_id}.%(ext)s")
        opts = {
            "format": "bestaudio/best/ba/b",
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
            "remote_components": ["ejs:github"],
            "js_runtimes": {"node": {}} if shutil.which("node") else {}
        }
        cookie_path = self._find_cookie_file()
        if cookie_path:
            opts["cookiefile"] = cookie_path
        return opts

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
        last_error_msg = ""

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
            last_error_msg = str(dl_err)
            log_error(f"Предупреждение загрузчика: {dl_err}")
            raw_path = self._try_direct_or_stream_download(clean_url, task_id)

        if not raw_path or not os.path.exists(raw_path):
            for candidate in self.output_dir.glob(f"{task_id}.*"):
                if candidate.is_file() and not candidate.name.endswith((".wav", ".part")):
                    raw_path = str(candidate)
                    break

        if not raw_path or not os.path.exists(raw_path):
            err_lower = last_error_msg.lower()
            if "confirm your age" in err_lower or "age-restricted" in err_lower or "inappropriate" in err_lower:
                log_info("Видео помечено как 18+ (возрастное ограничение). Для решения: установите расширение 'Get cookies.txt LOCALLY' для своего браузера, зайдите на YouTube, экспортируйте файл, назовите его cookies.txt и положите в папку проекта.")
                raise RuntimeError(
                    "Видео помечено как 18+ (возрастное ограничение). Для решения: установите расширение 'Get cookies.txt LOCALLY' для своего браузера, зайдите на YouTube, экспортируйте файл, назовите его cookies.txt и положите в папку проекта."
                )
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
