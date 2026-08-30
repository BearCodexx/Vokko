import os
import uuid
import shutil
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

from backend.app.core.config import UPLOAD_DIR, PROCESSED_DIR, EXPORTS_DIR
from backend.app.core.downloader import MediaDownloader
from backend.app.core.audio_utils import convert_to_wav
from backend.app.core.logger import log_info, log_stage, log_warning, log_error
from backend.app.core.progress import set_task_progress, get_task_progress, clear_task_progress
from backend.app.core.models_manager import (
    get_all_models_catalog,
    download_model_file,
    get_download_progress,
    load_app_config,
    save_app_config
)
from backend.app.services.whisper_engine import WhisperEngine
from backend.app.services.demucs_engine import DemucsEngine
from backend.app.services.diarization_engine import DiarizationEngine
from backend.app.services.lyric_analyzer import LyricAnalyzer
from backend.app.services.llm_engine import LLMEngine
from backend.app.services.export_engine import ExportEngine

router = APIRouter()

downloader = MediaDownloader()
whisper_srv = WhisperEngine()
demucs_srv = DemucsEngine()
diarizer = DiarizationEngine()
lyric_srv = LyricAnalyzer()
llm_srv = LLMEngine()
exporter = ExportEngine()

class SpeakerActionPayload(BaseModel):
    action: str
    segments: List[Dict[str, Any]]
    old_name: Optional[str] = None
    new_name: Optional[str] = None
    source_speaker: Optional[str] = None
    target_speaker: Optional[str] = None
    speaker_name: Optional[str] = None

class ExportPayload(BaseModel):
    export_format: str
    mode: str
    data: Dict[str, Any]
    filename: Optional[str] = "vokko_transcript"

class ActiveModelPayload(BaseModel):
    active_asr: Optional[str] = None
    active_llm: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None

# функция очистки временных файлов задачи
def _cleanup_task_files(task_id: str):
    try:
        for p in UPLOAD_DIR.glob(f"{task_id}*"):
            if p.name.endswith(".gitkeep"):
                continue
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass
    try:
        for p in PROCESSED_DIR.glob(f"*{task_id}*"):
            if p.name.endswith(".gitkeep"):
                continue
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass
    try:
        demucs_dir = UPLOAD_DIR.parent / "demucs_out" / f"demucs_{task_id}"
        if demucs_dir.exists():
            shutil.rmtree(demucs_dir, ignore_errors=True)
    except Exception:
        pass
    clear_task_progress(task_id)

# функция удаления сформированного файла экспорта после передачи
def _cleanup_export_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

# очистка остаточных временных файлов при старте
def _cleanup_all_temp_storage():
    for target in [UPLOAD_DIR, PROCESSED_DIR, EXPORTS_DIR, UPLOAD_DIR.parent / "demucs_out"]:
        try:
            if target.exists():
                for item in target.iterdir():
                    try:
                        if item.name.endswith(".gitkeep") or item.name == ".gitkeep":
                            continue
                        if item.is_file():
                            item.unlink(missing_ok=True)
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception:
                        pass
        except Exception:
            pass

_cleanup_all_temp_storage()

# получение каталога моделей с их статусом для выбранного движка
@router.get("/models")
async def get_models(engine: Optional[str] = None):
    return JSONResponse(content=get_all_models_catalog(engine=engine))

# запуск фонового скачивания выбранной модели
@router.post("/models/download")
async def start_model_download_endpoint(payload: Dict[str, str] = Body(...)):
    model_id = payload.get("model_id")
    engine = payload.get("engine", "faster-whisper")
    if not model_id:
        raise HTTPException(status_code=400, detail="Не указан идентификатор модели")

    threading.Thread(target=download_model_file, args=(model_id, engine), daemon=True).start()
    return JSONResponse(content={"status": "started", "model_id": model_id})

# опрос прогресса загрузки модели
@router.get("/models/progress/{model_id}")
async def check_model_download_progress(model_id: str, engine: Optional[str] = None):
    return JSONResponse(content=get_download_progress(model_id, engine=engine or "faster-whisper"))

# сохранение активных моделей
@router.post("/models/active")
async def set_active_models(payload: ActiveModelPayload):
    cfg = load_app_config()
    if payload.active_asr:
        cfg["active_asr_model"] = payload.active_asr
    if payload.active_llm:
        cfg["active_llm_model"] = payload.active_llm
    if payload.ollama_url:
        cfg["ollama_url"] = payload.ollama_url
    if payload.ollama_model:
        cfg["ollama_model"] = payload.ollama_model
    if payload.openrouter_api_key is not None:
        cfg["openrouter_api_key"] = payload.openrouter_api_key
    if payload.openrouter_model is not None:
        cfg["openrouter_model"] = payload.openrouter_model
    save_app_config(cfg)
    return JSONResponse(content={"status": "success", "config": cfg})

# опрос актуального состояния выполнения этапов задачи без блокировки очереди
@router.get("/progress/{task_id}")
async def get_progress_status(task_id: str):
    return JSONResponse(content=get_task_progress(task_id))

# получение списка доступных моделей на локальном сервере ollama
@router.get("/ollama/models")
async def get_ollama_models_list():
    cfg = load_app_config()
    base_url = cfg.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
    models = llm_srv._get_available_ollama_models(base_url)
    return JSONResponse(content={"models": models, "active": cfg.get("ollama_model", "")})

from backend.app.core.audio_utils import convert_to_wav, get_audio_duration

# обработка видео и звукозаписи в отдельном рабочем потоке
@router.post("/transcribe/general")
def transcribe_general(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    task_id: Optional[str] = Form(None),
    enable_timecodes: bool = Form(True),
    enable_diarization: bool = Form(True),
    num_speakers: Optional[int] = Form(None),
    asr_model: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    llm_layers: int = Form(1),
    batch_minutes: float = Form(8.0),
    whisper_engine_type: str = Form("faster-whisper"),
    beam_size: int = Form(1)
):
    current_task_id = task_id or str(uuid.uuid4())[:8]
    audio_path = None
    title = "Аудиозапись"
    duration = 0.0

    cfg = load_app_config()
    target_asr = asr_model or cfg.get("active_asr_model", "large-v3")
    target_llm = llm_model or cfg.get("active_llm_model", "none")
    total_stages = 2 + (1 if enable_diarization else 0) + (1 if target_llm != "none" else 0)
    current_stage = 1

    try:
        set_task_progress(current_task_id, current_stage, total_stages, "Загрузка аудио")
        log_stage(current_stage, total_stages, "Загрузка аудио")
        if file and file.filename:
            title = Path(file.filename).stem
            ext = Path(file.filename).suffix or ".mp3"
            log_info(f"Загрузка локального файла: {file.filename}")
            saved_file = UPLOAD_DIR / f"{current_task_id}{ext}"
            with open(saved_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            audio_path = convert_to_wav(str(saved_file))
        elif url and url.strip():
            log_info(f"Скачивание звукового потока по ссылке: {url.strip()}")
            dl_res = downloader.download_url(url.strip())
            audio_path = dl_res["file_path"]
            title = dl_res.get("title", title)
            duration = dl_res.get("duration", 0.0)
            log_info(f"Поток успешно загружен: {title}")
        else:
            raise HTTPException(status_code=400, detail="Необходимо предоставить файл или интернет адрес")

        # распознавание речи выбранным движком и моделью
        current_stage += 1
        set_task_progress(current_task_id, current_stage, total_stages, f"Распознавание речи Whisper ({target_asr})")
        log_stage(current_stage, total_stages, f"Распознавание речи Whisper ({target_asr}, движок: {whisper_engine_type}, beam: {beam_size})")
        whisper_res = whisper_srv.transcribe(
            audio_path,
            is_music=False,
            model_name=target_asr,
            engine_type=whisper_engine_type,
            beam_size=beam_size,
            task_id=current_task_id
        )
        if whisper_res.get("status") == "error":
            raise HTTPException(status_code=500, detail=whisper_res.get("error", "Ошибка транскрибации"))

        segments = whisper_res.get("segments", [])
        log_info(f"Распознано сегментов: {len(segments)}, язык: {whisper_res.get('language', 'ru')}")

        # выделение спикеров ДО нейрокоррекции для передачи контекста диалога
        if enable_diarization and len(segments) > 0:
            current_stage += 1
            set_task_progress(current_task_id, current_stage, total_stages, "Разделение спикеров")
            log_stage(current_stage, total_stages, "Разделение спикеров")
            segments = diarizer.diarize(audio_path, segments, num_speakers=num_speakers)

        # нейрокоррекция пунктуации и грамматики с учетом определенных спикеров
        if target_llm != "none":
            current_stage += 1
            set_task_progress(current_task_id, current_stage, total_stages, f"Коррекция текста ({target_llm})")
            log_stage(current_stage, total_stages, f"Коррекция текста моделью ({target_llm})")
            segments = llm_srv.correct_general_text(target_llm, segments, layers=llm_layers, batch_minutes=batch_minutes)

        if not enable_timecodes:
            for s in segments:
                s.pop("start_str", None)
                s.pop("end_str", None)

        return {
            "status": "success",
            "task_id": current_task_id,
            "title": title,
            "duration": duration or whisper_res.get("duration", 0.0),
            "language": whisper_res.get("language", "ru"),
            "full_text": whisper_res.get("text", ""),
            "segments": segments,
            "has_speakers": enable_diarization
        }
    except HTTPException as http_err:
        log_error(f"Ошибка запроса: {http_err.detail}")
        raise
    except Exception as err:
        log_error(f"Сбой при обработке: {str(err)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки, {str(err)}")
    finally:
        _cleanup_task_files(current_task_id)

# транскрибация музыкального трека в отдельном рабочем потоке
@router.post("/transcribe/music")
def transcribe_music(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    task_id: Optional[str] = Form(None),
    asr_model: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    llm_layers: int = Form(1)
):
    current_task_id = task_id or str(uuid.uuid4())[:8]
    audio_path = None
    title = "Музыкальная дорожка"
    artist = ""
    duration = 0.0

    cfg = load_app_config()
    target_asr = asr_model or cfg.get("active_asr_model", "large-v3")
    target_llm = llm_model or cfg.get("active_llm_model", "none")

    try:
        set_task_progress(current_task_id, 1, 4, "Загрузка аудио")
        log_stage(1, 4, "Загрузка аудио")
        if file and file.filename:
            title = Path(file.filename).stem
            ext = Path(file.filename).suffix or ".mp3"
            log_info(f"Загрузка музыкального файла: {file.filename}")
            saved_file = UPLOAD_DIR / f"{current_task_id}{ext}"
            with open(saved_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            audio_path = convert_to_wav(str(saved_file))
        elif url and url.strip():
            log_info(f"Загрузка музыкального трека по адресу: {url.strip()}")
            dl_res = downloader.download_url(url.strip())
            audio_path = dl_res["file_path"]
            title = dl_res.get("title", title)
            artist = dl_res.get("artist", "")
            duration = dl_res.get("duration", 0.0)
            log_info(f"Трек загружен: {artist} - {title}")
        else:
            raise HTTPException(status_code=400, detail="Необходимо предоставить файл или ссылку на песню")

        # проверка ограничения длительности трека для музыкального режима
        actual_duration = duration or get_audio_duration(audio_path)
        if actual_duration > 480.0:
            log_error(f"Длительность трека ({actual_duration:.1f} сек) превышает 8 минут")
            raise HTTPException(
                status_code=400,
                detail="Длительность трека для музыкального режима ограничена 8 минутами (480 сек)"
            )

        # изоляция вокала через выделение дорожек
        set_task_progress(current_task_id, 2, 4, "Разделение вокала и музыки (Demucs)")
        log_stage(2, 4, "Разделение вокала и музыки (Demucs)")
        vocal_audio_path = demucs_srv.isolate_vocals(audio_path, current_task_id)

        # распознавание текста песни выбранной моделью
        set_task_progress(current_task_id, 3, 4, f"Распознавание текста песни (Whisper)")
        log_stage(3, 4, f"Распознавание текста песни (Whisper {target_asr})")
        lyric_prompt = f"Песня {title} {artist}".strip()
        whisper_res = whisper_srv.transcribe(vocal_audio_path, prompt=lyric_prompt, is_music=True, model_name=target_asr)
        if whisper_res.get("status") == "error":
            raise HTTPException(status_code=500, detail=whisper_res.get("error", "Ошибка транскрибации вокала"))

        segments = whisper_res.get("segments", [])

        # интеллектуальный анализ куплетов и припевов
        set_task_progress(current_task_id, 4, 4, "Разметка куплетов и припевов (LLM)")
        log_stage(4, 4, f"Разметка куплетов и припевов ({target_llm})")
        blocks = lyric_srv.analyze_structure(segments)

        # опциональная нейрокоррекция и выравнивание строф через LLM
        if target_llm != "none":
            blocks = llm_srv.correct_song_lyrics(target_llm, segments, blocks, song_title=title, song_artist=artist, layers=llm_layers)

        log_info(f"Сформировано смысловых блоков песни: {len(blocks)}")

        return {
            "status": "success",
            "task_id": current_task_id,
            "title": title,
            "artist": artist,
            "duration": duration or whisper_res.get("duration", 0.0),
            "language": whisper_res.get("language", "ru"),
            "blocks": blocks,
            "segments": segments,
            "full_text": whisper_res.get("text", "")
        }
    except HTTPException as http_err:
        log_error(f"Ошибка обработки: {http_err.detail}")
        raise
    except Exception as err:
        log_error(f"Сбой при обработке песни: {str(err)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки песни, {str(err)}")
    finally:
        _cleanup_task_files(current_task_id)

# переименование, слияние и удаление спикеров
@router.post("/speakers/action")
async def handle_speakers_action(payload: SpeakerActionPayload):
    action = payload.action.lower()
    segments = payload.segments

    if action == "rename":
        if not payload.old_name or not payload.new_name:
            raise HTTPException(status_code=400, detail="Не указаны имена для переименования")
        segments = diarizer.rename_speaker(segments, payload.old_name, payload.new_name)
    elif action == "merge":
        if not payload.source_speaker or not payload.target_speaker:
            raise HTTPException(status_code=400, detail="Не указаны участники для слияния")
        segments = diarizer.merge_speakers(segments, payload.source_speaker, payload.target_speaker)
    elif action == "delete":
        if not payload.speaker_name:
            raise HTTPException(status_code=400, detail="Не указан спикер для удаления")
        segments = diarizer.delete_speaker(segments, payload.speaker_name)
    else:
        raise HTTPException(status_code=400, detail="Неизвестная операция со спикерами")

    return JSONResponse(content={"status": "success", "segments": segments})

# экспорт готовой расшифровки в файл нужного формата
@router.post("/export")
async def export_transcript(payload: ExportPayload):
    fmt = payload.export_format.lower()
    mode = payload.mode.lower()
    data = payload.data
    fname = payload.filename or "vokko_transcript"

    out_path = None
    media_type = "text/plain"

    try:
        if fmt == "txt":
            out_path = exporter.export_txt(fname, data, mode=mode)
            media_type = "text/plain; charset=utf-8"
        elif fmt == "docx":
            out_path = exporter.export_docx(fname, data, mode=mode)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif fmt == "pdf":
            out_path = exporter.export_pdf(fname, data, mode=mode)
            media_type = "application/pdf"
        elif fmt == "srt":
            out_path = exporter.export_srt(fname, data)
            media_type = "text/plain; charset=utf-8"
        elif fmt == "lrc":
            out_path = exporter.export_lrc(fname, data)
            media_type = "text/plain; charset=utf-8"
        elif fmt == "json":
            out_path = exporter.export_json(fname, data)
            media_type = "application/json"
        else:
            raise HTTPException(status_code=400, detail="Формат выгрузки не поддерживается")

        if not out_path or not os.path.exists(out_path):
            raise HTTPException(status_code=500, detail="Ошибка генерации документа")

        return FileResponse(
            path=out_path,
            filename=os.path.basename(out_path),
            media_type=media_type,
            background=BackgroundTask(_cleanup_export_file, out_path)
        )
    except Exception as err:
        log_error(f"Сбой экспорта: {str(err)}")
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(err)}")
