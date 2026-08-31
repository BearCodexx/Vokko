import os
import sys
import re
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.app.core.config import MODELS_LLM_DIR
from backend.app.core.models_manager import get_model_path, load_app_config, LLM_MODELS
from backend.app.core.logger import log_info, log_error

# цветовая палитра для динамического прогресс-бара
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_MAGENTA = "\033[95m"
CLR_DIM = "\033[90m"

# отрисовка бегущей строки прогресса нейрокоррекции в одну строку
def _render_live_llm_progress_bar(current_batch: int, total_batches: int, time_range: str, is_final: bool = False):
    bar_len = 26
    pct = min(100.0, (current_batch / max(1, total_batches)) * 100.0)
    filled = int(bar_len * (pct / 100.0))
    bar = f"{CLR_GREEN}{'=' * filled}{CLR_DIM}{'-' * (bar_len - filled)}{CLR_RESET}"
    sys.stdout.write(f"\r  [{bar}] {CLR_CYAN}{pct:5.1f}%{CLR_RESET} (Батч {current_batch}/{total_batches}) | {CLR_MAGENTA}{time_range}{CLR_RESET}   ")
    sys.stdout.flush()
    if is_final:
        sys.stdout.write("\n")
        sys.stdout.flush()

# сервис нейрокоррекции текста, пунктуации и структуры с помощью языковых моделей
class LLMEngine:

    def __init__(self):
        self._current_model = None
        self._current_model_id = None

    # вызов локальной модели через библиотеку инференса
    def _call_llama_cpp(self, model_path: Path, prompt: str, system_prompt: str, temperature: float = 0.25) -> Optional[str]:
        try:
            import llama_cpp
        except ImportError:
            return None

        try:
            if self._current_model_id != str(model_path):
                self._current_model = llama_cpp.Llama(
                    model_path=str(model_path),
                    n_gpu_layers=0,
                    n_ctx=4096,
                    verbose=False
                )
                self._current_model_id = str(model_path)

            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            output = self._current_model(
                full_prompt,
                max_tokens=2048,
                temperature=temperature,
                top_p=0.9,
                stop=["<|im_end|>", "</s>"]
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            log_error(f"Ошибка выполнения инференса локальной модели: {str(e)}")
            return None

    # получение списка доступных моделей на сервере ollama
    def _get_available_ollama_models(self, base_url: str) -> List[str]:
        try:
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                return [m for m in models if "embed" not in m.lower()]
        except Exception:
            return []

    # вызов облачного сервиса через интерфейс openrouter
    def _call_openrouter(self, prompt: str, system_prompt: str, temperature: float = 0.25, timeout: int = 300) -> Optional[str]:
        cfg = load_app_config()
        api_key = cfg.get("openrouter_api_key", "").strip()
        model_name = cfg.get("openrouter_model", "google/gemma-2-9b-it").strip() or "google/gemma-2-9b-it"

        if not api_key:
            log_error("Ключ доступа к OpenRouter не указан в настройках")
            return None

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }

        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://vokko.app",
                    "X-Title": "Vokko Transcription"
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ans = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return ans
        except Exception as e:
            log_error(f"Сбой обращения к OpenRouter: {e}")
            return None

    # вызов сервера ollama с точным сопоставлением выбранной модели
    def _call_ollama(self, prompt: str, system_prompt: str, model_id: str = "", temperature: float = 0.25, timeout: int = 900) -> Optional[str]:
        cfg = load_app_config()
        base_url = cfg.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
        configured_model = cfg.get("ollama_model", "")

        available_models = self._get_available_ollama_models(base_url)
        if not available_models:
            log_error(f"Сервер Ollama не отвечает или нет установленных моделей по адресу {base_url}")
            return None

        # поиск точного или наиболее близкого имени модели
        target_model = None
        if model_id:
            clean_id = model_id.replace(".gguf", "").lower()
            for m in available_models:
                m_clean = m.lower().replace(":", "-")
                if clean_id in m_clean or m_clean in clean_id or model_id.lower() in m.lower():
                    target_model = m
                    break

        if not target_model and configured_model in available_models:
            target_model = configured_model

        if not target_model:
            general_llms = [m for m in available_models if "llama" in m.lower() or "qwen" in m.lower() or "gemma" in m.lower()]
            target_model = general_llms[0] if general_llms else available_models[0]

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            # Динамический таймаут на основе длительности аудио
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ans = data.get("message", {}).get("content", "").strip()
                return ans
        except Exception as e:
            err_str = str(e).lower()
            if "time" in err_str or "out" in err_str:
                log_error(f"Таймаут ожидания локальной модели ({timeout} сек), повторная попытка...")
            else:
                log_error(f"Сбой обращения к Ollama: {e}")
            return None

    # единая точка вызова генерации ответа в зависимости от выбранной модели
    def generate(self, model_id: str, prompt: str, system_prompt: str, temperature: float = 0.25, timeout: int = 900) -> Optional[str]:
        if model_id == "none":
            return None

        if model_id == "openrouter":
            return self._call_openrouter(prompt, system_prompt, temperature=temperature, timeout=timeout)

        if model_id == "custom-ollama":
            return self._call_ollama(prompt, system_prompt, model_id=model_id, temperature=temperature, timeout=timeout)

        # приоритетное обращение к зарегистрированной модели
        res = self._call_ollama(prompt, system_prompt, model_id=model_id, temperature=temperature, timeout=timeout)
        if res:
            return res

        model_path = get_model_path(model_id)
        if model_path and model_path.exists():
            return self._call_llama_cpp(model_path, prompt, system_prompt, temperature=temperature)

        return None

    # универсальное распознавание заголовков строф на любых языках
    @staticmethod
    def _match_block_header(line: str) -> Optional[str]:
        t = line.strip()
        pattern = r'^\s*\[?\s*([KКkк][Уу][Пп][Лл][Ее][Тт]|[Пп][Рр][Ии][Пп][Ее][Вв]|[Бб][Рр][Ии][Дд][Жж]|[Аа][Уу][Тт][Рр][Оо]|[Ии][Нн][Тт][Рр][Оо]|[Vv][Ee][Rr][Ss][Ee]|[Cc][Hh][Oo][Rr][Uu][Ss]|[Bb][Rr][Ii][Dd][Gg][Ee]|[Oo][Uu][Tt][Rr][Oo]|[Ii][Nn][Tt][Rr][Oo])\s*(\d*)\s*\]?[:\.]?\s*$'
        m = re.match(pattern, t, re.IGNORECASE)
        if m:
            raw_title = m.group(1).lower()
            num = m.group(2) or ""
            if any(p in raw_title for p in ["купл", "verse"]):
                return f"Куплет {num}".strip() if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in raw_title) else f"Verse {num}".strip()
            elif any(p in raw_title for p in ["прип", "chorus"]):
                return f"Припев {num}".strip() if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in raw_title) else f"Chorus {num}".strip()
            elif any(p in raw_title for p in ["бридж", "bridge"]):
                return f"Бридж {num}".strip() if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in raw_title) else f"Bridge {num}".strip()
            elif any(p in raw_title for p in ["аутро", "outro"]):
                return "Аутро" if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in raw_title) else "Outro"
            elif any(p in raw_title for p in ["интро", "intro"]):
                return "Интро" if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in raw_title) else "Intro"
            return f"Куплет {num}".strip()
        return None

    # универсальное разделение составных строк без разрыва поэтических фраз
    @staticmethod
    def _split_compound_line(line: str) -> List[str]:
        cleaned = line.strip()
        if not cleaned:
            return []
        chunks = re.split(r'[\r\n]+', cleaned)
        res = [c.strip() for c in chunks if c.strip()]
        return res if res else [cleaned]

    # универсальная нейрокоррекция текста песни с поддержкой температурных слоев
    def correct_song_lyrics(
        self,
        model_id: str,
        segments: List[Dict[str, Any]],
        blocks: List[Dict[str, Any]],
        song_title: str = "",
        song_artist: str = "",
        layers: int = 1
    ) -> List[Dict[str, Any]]:
        if model_id == "none" or not blocks:
            return blocks

        # сборка чистого сырого текста для передачи в модель
        all_raw_lines = []
        for b in blocks:
            for l in b.get("lines", []):
                for sub in self._split_compound_line(l.get("text", "")):
                    all_raw_lines.append(sub)

        input_payload = "\n".join(all_raw_lines)
        meta_info = f"Track: {song_artist} - {song_title}" if song_title else "Track"

        # строгий системный промпт с четким определением структуры песни и обязательной пунктуацией
        system_prompt = (
            "You are a master song lyricist and poetic reconstruction editor.\n"
            f"Context: {meta_info}.\n"
            "The input is raw unpunctuated ASR transcription from an audio vocal track. Due to distortion, instruments, and singing timbre, it contains phonetic errors, chopped lines, and missing punctuation.\n\n"
            "CRITICAL SONG RECONSTRUCTION RULES:\n"
            "1. LANGUAGE PRESERVATION:\n"
            "   - Always output in the EXACT original language of the song (e.g. Russian in Russian, English in English). NEVER translate lyrics.\n"
            "2. STANZA STRUCTURE & HEADERS:\n"
            "   - [Куплет 1] / [Verse 1]: First unique storytelling verse (4-8 lines).\n"
            "   - [Припев 1] / [Chorus 1]: Main repeating emotional hook.\n"
            "   - [Куплет 2] / [Verse 2]: Second unique verse (4-8 lines).\n"
            "   - [Припев 2] / [Chorus 2]: EXACT IDENTICAL repetition of Chorus 1 lyrics.\n"
            "   - [Бридж] / [Bridge]: Optional transitional unique section.\n"
            "   - [Аутро] / [Outro]: Ending lines.\n"
            "3. RHYME & PHONETIC RESTORATION:\n"
            "   - Correct acoustic mishearings into true poetic words that rhyme and fit the musical meter.\n"
            "4. COMPLETE POETIC LINES:\n"
            "   - Every line must be a complete musical phrase (4 to 8 words per line). Never break lines into 1-2 words.\n"
            "5. STRICT PUNCTUATION & CAPITALIZATION:\n"
            "   - Capitalize the first letter of each line.\n"
            "   - Restore complete grammatical punctuation: commas at phrase breaks, periods at stanza ends, dashes, question marks.\n"
            "6. RAW OUTPUT ONLY:\n"
            "   - Output ONLY the polished lyrics starting directly with [Куплет 1]. NEVER output introductions, bullet points, explanations, or changelogs."
        )

        user_prompt = f"Restore and polish the song into standard verses and choruses with full punctuation:\n\n{input_payload}"

        # распределение температурных режимов в зависимости от числа слоев
        temp_profiles = {
            1: [0.20],
            2: [0.20, 0.40],
            3: [0.15, 0.35, 0.55],
            4: [0.10, 0.25, 0.45, 0.65],
            5: [0.05, 0.20, 0.35, 0.50, 0.70]
        }
        temps = temp_profiles.get(max(1, min(5, layers)), [0.20])

        log_info(f"Запуск нейрокоррекции текста песни моделью: {model_id} (слоев: {len(temps)})")

        if len(temps) == 1:
            corrected_response = self.generate(model_id, user_prompt, system_prompt, temperature=temps[0])
        else:
            candidates = []
            for t_val in temps:
                ans = self.generate(model_id, user_prompt, system_prompt, temperature=t_val)
                if ans and len(ans.strip()) > 30 and ans not in candidates:
                    candidates.append(ans)

            if not candidates:
                corrected_response = None
            elif len(candidates) == 1:
                corrected_response = candidates[0]
            else:
                # синтез идеального варианта из нескольких температурных проходов
                synthesis_system = (
                    "You are a master song lyricist and consensus judge.\n"
                    "RULES:\n"
                    "1. Always output in the EXACT same language as the song (e.g. Russian in Russian). Never translate.\n"
                    "2. Combine the best elements into standard stanzas ([Куплет 1], [Припев 1], [Куплет 2], [Припев 2], [Бридж], [Аутро]).\n"
                    "3. Ensure repeated choruses ([Припев 1], [Припев 2]) have EXACTLY identical lyrics.\n"
                    "4. Restore full grammatical punctuation: capitalize every line, place commas at line breaks, periods at stanza ends.\n"
                    "5. Output ONLY the final lyrics starting directly with [Куплет 1]. ABSOLUTELY NO commentary, notes, or bullet points."
                )
                synthesis_input = "\n\n=== VARIANT CANDIDATES ===\n\n" + "\n\n---\n\n".join(f"VARIANT {idx+1}:\n{c}" for idx, c in enumerate(candidates))
                log_info(f"Синтез идеального варианта из {len(candidates)} слоев нейрокоррекции")
                corrected_response = self.generate(model_id, synthesis_input, synthesis_system, temperature=0.10)

        if not corrected_response or len(corrected_response.strip()) < 20:
            log_info("Нейрокоррекция не вернула результат, сохранена базовая структура")
            return blocks

        # парсинг ответа модели обратно в блоки строф с фильтрацией отсебятины
        parsed_blocks = []
        cur_b = None

        filter_phrases = [
            "после анализа", "после тщательного", "я предлагаю", "версия песни", "в этом варианте",
            "я сделал следующие", "в результате получилась", "соответствует рифме", "исправил порядок",
            "оставил исходный", "подходит по рифме", "выведи только", "исправь ошибки", "исправленный текст",
            "текст песни:", "заголовками строф", "вот исправленный", "here is the", "restored lyrics:"
        ]

        for raw_line in corrected_response.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # фильтрация маркдаун списков изменений (* В куплете..., - В припеве...)
            if line.startswith(("*", "-", "•", "#")) and any(p in line.lower() for p in ["куплет", "припев", "вариант", "строк", "изменен", "исправ"]):
                continue

            # фильтрация вводных/заключительных фраз модели
            if any(p in line.lower() for p in filter_phrases):
                continue

            # проверка заголовка строфы
            matched_title = self._match_block_header(line)
            if matched_title:
                if cur_b and cur_b["lines"]:
                    parsed_blocks.append(cur_b)
                b_type = "chorus" if any(p in matched_title.lower() for p in ["припев", "chorus"]) else "verse"
                cur_b = {
                    "id": len(parsed_blocks) + 1,
                    "type": b_type,
                    "title": matched_title,
                    "lines": []
                }
            else:
                # если заголовок еще не встретился, не создаем ложный блок из вступительного текста
                if cur_b is None:
                    # проверяем, похожа ли строка на стихотворную строчку
                    if len(line.split()) > 10 and not any(p in line.lower() for p in ["песн", "весна", "небо", "стена", "дождь", "ночь"]):
                        continue
                    cur_b = {"id": 1, "type": "verse", "title": "Куплет 1", "lines": []}

                for sub in self._split_compound_line(line):
                    sub_clean = sub.strip()
                    if sub_clean:
                        cur_b["lines"].append({
                            "time_str": "",
                            "text": sub_clean
                        })

        if cur_b and cur_b["lines"]:
            parsed_blocks.append(cur_b)

        if not parsed_blocks:
            return blocks

        log_info(f"Нейрокоррекция успешно выполнена, сформировано блоков: {len(parsed_blocks)}")
        return parsed_blocks

    # разбивка сегментов речи на временные батчи с учетом завершения реплик спикеров
    def _split_into_smart_batches(self, segments: List[Dict[str, Any]], target_duration_sec: float) -> List[List[Dict[str, Any]]]:
        if not segments:
            return []

        batches = []
        current_batch = []
        batch_start_time = float(segments[0].get("start", 0.0))
        max_duration_sec = target_duration_sec * 1.15

        for i, seg in enumerate(segments):
            if not current_batch:
                current_batch.append(seg)
                batch_start_time = float(seg.get("start", 0.0))
                continue

            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            batch_elapsed = seg_end - batch_start_time
            is_last = (i == len(segments) - 1)
            next_seg = segments[i + 1] if not is_last else None

            # проверка естественного завершения мысли (точка, пауза или смена спикера)
            txt = seg.get("text", "").strip()
            ends_sentence = any(txt.endswith(p) for p in [".", "!", "?", "...", "»", '"'])
            pause_to_next = (float(next_seg.get("start", 0.0)) - seg_end) if next_seg else 999.0
            speaker_changed = bool(seg.get("speaker") and next_seg and next_seg.get("speaker") != seg.get("speaker"))

            should_cut = False
            if batch_elapsed >= target_duration_sec:
                if ends_sentence or pause_to_next >= 0.8 or speaker_changed or batch_elapsed >= max_duration_sec:
                    should_cut = True

            current_batch.append(seg)
            if should_cut and not is_last:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        return batches

    # универсальная грамматическая коррекция речи для стенограмм с нарезкой по батчам
    def correct_general_text(
        self,
        model_id: str,
        segments: List[Dict[str, Any]],
        layers: int = 1,
        batch_minutes: float = 8.0
    ) -> List[Dict[str, Any]]:
        if model_id == "none" or not segments:
            return segments

        # расчет длительности одного батча в секундах
        target_sec = max(60.0, float(batch_minutes) * 60.0)
        batches = self._split_into_smart_batches(segments, target_sec)
        total_batches = len(batches)

        log_info(f"Запуск нейрокоррекции речи моделью: {model_id} (батчей: {total_batches}, размер батча: {batch_minutes} мин)")

        system_prompt = (
            "Ты — профессиональный редактор и корректор стенограмм устной речи.\n"
            "Тебе передан пронумерованный список реплик из аудиозаписи.\n"
            "ЗАДАЧИ:\n"
            "1. СТРОГО СОХРАНЯЙ ИСХОДНЫЙ ЯЗЫК ОРИГИНАЛА (русский в русском, английский в английском и т.д.). КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО ПЕРЕВОДИТЬ ТЕКСТ НА ДРУГОЙ ЯЗЫК.\n"
            "2. Расставь правильную пунктуацию (запятые, точки, тире, вопросительные знаки) и заглавные буквы.\n"
            "3. Удали слова-паразиты и заикания (э-э, ну, как бы, вот, эм / um, uh, like).\n"
            "4. Исправь фонетические ослышки транскрибатора по смыслу фразы.\n"
            "5. ОБЯЗАТЕЛЬНО сохраняй номера строк вида '[1] ...', '[2] ...' в начале каждой строки.\n"
            "6. Выводи ровно по одной строке на каждый номер в точном порядке без пропусков, объединений и комментариев от себя."
        )

        _render_live_llm_progress_bar(0, total_batches, "00:00 - ...", is_final=False)

        for b_idx, batch_segs in enumerate(batches, 1):
            start_str = batch_segs[0].get("start_str") or "00:00"
            end_str = batch_segs[-1].get("end_str") or batch_segs[-1].get("start_str") or "00:00"
            
            # Таймаут равен длительности батча (минимум 60 секунд)
            batch_dur_sec = max(60, int(float(batch_segs[-1].get("end", 0.0)) - float(batch_segs[0].get("start", 0.0))))

            # вспомогательная функция обработки пачки сегментов
            def _process_subgroup(sub_segs):
                lines_payload = []
                for idx, s in enumerate(sub_segs, 1):
                    txt = s.get("text", "").strip()
                    lines_payload.append(f"[{idx}] {txt}")

                prompt_text = "\n".join(lines_payload)
                if not prompt_text.strip():
                    return

                ans = None
                # Пользователь просил 1 повтор при неудаче (то есть всего 2 попытки)
                for _attempt in range(2):
                    ans = self.generate(model_id, prompt_text, system_prompt, temperature=0.15, timeout=batch_dur_sec)
                    if ans:
                        break
                    time.sleep(2)
                    
                if not ans:
                    return

                # фильтрация отказов безопасности и служебных отписок
                ans_lower = ans.lower()
                refusal_markers = [
                    "cannot create", "i am sorry", "i apologize", "as an ai", "language model",
                    "не могу выполнять", "не могу выполнить", "не могу обработать", "не могу отвечать",
                    "извините", "извиняюсь", "как языковая модель", "как искусственный интеллект"
                ]
                if any(m in ans_lower for m in refusal_markers) and len(ans_lower) < 200:
                    return

                # двусторонняя защита от нежелательного перевода между языками
                prompt_cyr = len(re.findall(r'[а-яА-ЯёЁ]', prompt_text))
                prompt_lat = len(re.findall(r'[a-zA-Z]', prompt_text))
                ans_cyr = len(re.findall(r'[а-яА-ЯёЁ]', ans))
                ans_lat = len(re.findall(r'[a-zA-Z]', ans))

                # если исходник на русском, а ответ переведен на английский
                if prompt_cyr > 15 and prompt_cyr > prompt_lat:
                    if (ans_cyr / max(1, ans_cyr + ans_lat)) < 0.35:
                        return

                # если исходник на английском/латинице, а ответ переведен на русский
                if prompt_lat > 15 and prompt_cyr == 0:
                    if (ans_cyr / max(1, ans_cyr + ans_lat)) > 0.25:
                        return

                ans_lines = [l.strip() for l in ans.splitlines() if l.strip()]
                line_map = {}
                for l in ans_lines:
                    l_low = l.lower()
                    if any(ref in l_low for ref in ["не могу выполнять", "не могу выполнить", "извините", "извиняюсь", "as an ai", "i cannot"]):
                        continue

                    # строгое сопоставление по номеру строки
                    m = re.match(r'^\s*\[?(\d+)\]?[\.\:\)]?\s*(?:(?:Спикер|Speaker)\s*\d+[\:\.\-]?\s*)?(.*)$', l, re.IGNORECASE)
                    if m:
                        try:
                            num = int(m.group(1))
                            clean_body = m.group(2).strip()
                            clean_body = re.sub(r'^(?:Спикер|Speaker)\s*\d+[\:\.\-]?\s*', '', clean_body, flags=re.IGNORECASE).strip()
                            if clean_body:
                                line_map[num] = clean_body
                        except Exception:
                            pass

                # сопоставляем только строго по номеру
                for idx, s in enumerate(sub_segs, 1):
                    if idx in line_map:
                        mapped_text = line_map[idx]
                        orig_len = len(s.get("text", ""))
                        if orig_len == 0 or (0.25 <= len(mapped_text) / max(1, orig_len) <= 3.5):
                            s["text"] = mapped_text

            # обработка батча порциями по 45 сегментов для быстрой генерации без таймаутов
            chunk_size = 45
            for offset in range(0, len(batch_segs), chunk_size):
                chunk = batch_segs[offset:offset + chunk_size]
                _process_subgroup(chunk)

            _render_live_llm_progress_bar(b_idx, total_batches, f"{start_str} - {end_str}", is_final=(b_idx == total_batches))

        log_info(f"Нейрокоррекция завершена: 100% | Всего обработано батчей: {total_batches}")
        return segments
