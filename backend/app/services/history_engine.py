import os
import json
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.app.core.config import HISTORY_DIR
from backend.app.core.logger import log_info, log_error

# сервис управления историей последних десяти транскрипций
class HistoryEngine:
    def __init__(self, storage_dir: Path = HISTORY_DIR, max_items: int = 10):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self.index_file = self.storage_dir / "index.json"
        self._ensure_index()

    # проверка наличия индексного файла
    def _ensure_index(self):
        if not self.index_file.exists():
            self._write_index([])

    # чтение индекса сохраненных записей
    def _read_index(self) -> List[Dict[str, Any]]:
        try:
            if self.index_file.exists():
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    # запись обновленного индекса
    def _write_index(self, items: List[Dict[str, Any]]):
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log_error(f"Не удалось обновить индекс истории: {e}")

    # генерация краткого текста предварительного просмотра
    def _generate_preview(self, segments: List[Dict[str, Any]], blocks: List[Dict[str, Any]]) -> str:
        texts = []
        if segments:
            for s in segments[:4]:
                txt = s.get("text", "").strip()
                if txt:
                    texts.append(txt)
        elif blocks:
            for b in blocks[:3]:
                for l in b.get("lines", [])[:2]:
                    txt = l.get("text", "").strip()
                    if txt:
                        texts.append(txt)
        preview = " ".join(texts)
        if len(preview) > 180:
            preview = preview[:177] + "..."
        return preview

    # получение списка сохраненных транскрипций
    def list_history(self) -> List[Dict[str, Any]]:
        return self._read_index()

    # загрузка полных данных выбранной транскрипции
    def get_history_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        item_path = self.storage_dir / f"{item_id}.json"
        if not item_path.exists():
            return None
        try:
            with open(item_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log_error(f"Ошибка чтения записи истории {item_id}: {e}")
            return None

    # сохранение транскрипции с автоматической ротацией старых записей
    def save_history_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        item_id = str(data.get("id") or uuid.uuid4().hex[:12])
        title = data.get("title") or "Транскрипция"
        mode = data.get("mode") or "general"
        duration = float(data.get("duration") or 0.0)
        segments = data.get("segments") or []
        blocks = data.get("blocks") or []
        created_at = data.get("created_at") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        segments_count = len(segments) if segments else sum(len(b.get("lines", [])) for b in blocks)
        preview = self._generate_preview(segments, blocks)

        full_record = {
            "id": item_id,
            "title": title,
            "mode": mode,
            "duration": duration,
            "created_at": created_at,
            "segments_count": segments_count,
            "preview_text": preview,
            "segments": segments,
            "blocks": blocks,
            "artist": data.get("artist", ""),
            "metadata": data.get("metadata", {})
        }

        # сохраняем отдельный файл записи
        item_path = self.storage_dir / f"{item_id}.json"
        with open(item_path, "w", encoding="utf-8") as f:
            json.dump(full_record, f, ensure_ascii=False, indent=2)

        meta_record = {
            "id": item_id,
            "title": title,
            "mode": mode,
            "duration": duration,
            "created_at": created_at,
            "segments_count": segments_count,
            "preview_text": preview,
            "artist": data.get("artist", "")
        }

        index_items = self._read_index()
        # удаляем дубликат по id если уже существовал
        index_items = [it for it in index_items if it.get("id") != item_id]
        index_items.insert(0, meta_record)

        # автоматическая очистка записей сверх лимита
        if len(index_items) > self.max_items:
            for stale_item in index_items[self.max_items:]:
                stale_id = stale_item.get("id")
                if stale_id:
                    stale_file = self.storage_dir / f"{stale_id}.json"
                    try:
                        stale_file.unlink(missing_ok=True)
                    except Exception:
                        pass
            index_items = index_items[:self.max_items]

        self._write_index(index_items)
        return full_record

    # удаление конкретной транскрипции из истории
    def delete_history_item(self, item_id: str) -> bool:
        index_items = self._read_index()
        new_items = [it for it in index_items if it.get("id") != item_id]
        self._write_index(new_items)

        item_path = self.storage_dir / f"{item_id}.json"
        try:
            if item_path.exists():
                item_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            log_error(f"Не удалось удалить файл истории {item_id}: {e}")
            return False

    # полная очистка всей истории транскрипций
    def clear_history(self) -> bool:
        index_items = self._read_index()
        for it in index_items:
            stale_id = it.get("id")
            if stale_id:
                stale_file = self.storage_dir / f"{stale_id}.json"
                try:
                    stale_file.unlink(missing_ok=True)
                except Exception:
                    pass
        self._write_index([])
        return True

    # импорт внешней транскрипции из json
    def import_history_item(self, raw_data: Any) -> Dict[str, Any]:
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)
        if not isinstance(raw_data, dict):
            raise ValueError("Некорректный формат файла транскрипции")

        # поддержка импорта как стандартного vokko json, так и внешних форматов
        title = raw_data.get("title") or raw_data.get("filename") or "Импортированная запись"
        mode = raw_data.get("mode") or ("music" if raw_data.get("blocks") else "general")
        segments = raw_data.get("segments") or []
        blocks = raw_data.get("blocks") or []

        # если импортируется простой текст или список фраз
        if not segments and not blocks and "text" in raw_data:
            text = str(raw_data["text"]).strip()
            segments = [{"id": 1, "start": 0.0, "end": 0.0, "text": text, "speaker": "Спикер 1"}]

        prepared = {
            "title": title,
            "mode": mode,
            "duration": float(raw_data.get("duration") or 0.0),
            "segments": segments,
            "blocks": blocks,
            "artist": raw_data.get("artist", ""),
            "metadata": raw_data.get("metadata", {})
        }
        return self.save_history_item(prepared)
