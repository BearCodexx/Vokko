import backend.app.core.doh
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.api.routes import router as api_router
from backend.app.core.config import BASE_DIR

app = FastAPI(
    title="Vokko Transcription Platform",
    description="Продвинутая система транскрибации речи и музыки на базе Whisper Large v3 и Demucs",
    version="1.0.0"
)

# настройка кросс доменных запросов для работы интерфейса
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# подключение маршрутов обработки
app.include_router(api_router, prefix="/api")

# подключение статических файлов пользовательского интерфейса
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

# запуск при прямом вызове
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
