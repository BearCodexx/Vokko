import os
import sys
import json
import time
import urllib.request
from typing import Dict, Any, List
from backend.app.core.config import WIZARD_STATE_FILE, CONFIG_FILE
from backend.app.core.models_manager import (
    WHISPER_MODELS,
    LLM_MODELS,
    is_model_downloaded,
    download_model_file,
    load_app_config,
    save_app_config
)

# включение поддержки ansi цветов в консоли windows
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

# фирменная трехцветная палитра vokko
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_MAGENTA = "\033[95m"
CLR_DIM = "\033[90m"

# чтение текущего сохраненного состояния мастера настройки
def _load_wizard_state() -> Dict[str, Any]:
    default_state = {
        "status": "not_started",
        "step": "ask_start",
        "selected_asr": [],
        "selected_llm": [],
        "downloaded": []
    }
    if WIZARD_STATE_FILE.exists():
        try:
            with open(WIZARD_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_state.update(data)
        except Exception:
            pass
    return default_state

# сохранение состояния мастера в файл для продолжения при перезапуске
def _save_wizard_state(state: Dict[str, Any]):
    try:
        WIZARD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WIZARD_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# опрос доступных моделей на локальном сервере ollama
def _fetch_ollama_local_models() -> List[str]:
    cfg = load_app_config()
    base_url = cfg.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            return [m for m in models if "embed" not in m.lower()]
    except Exception:
        return []

# отрисовка фирменного цветного прогресс бара
def _cli_progress_bar(downloaded: int, total: int, pct: float, label: str):
    bar_len = 28
    filled = int(bar_len * (pct / 100.0))
    bar = f"{CLR_GREEN}{'=' * filled}{CLR_DIM}{'-' * (bar_len - filled)}{CLR_RESET}"
    mb_down = downloaded / (1024 * 1024)
    mb_tot = total / (1024 * 1024) if total > 0 else 0
    sys.stdout.write(f"\r  [{bar}] {CLR_CYAN}{pct:5.1f}%{CLR_RESET} ({mb_down:6.1f}/{mb_tot:6.1f} МБ) | {CLR_MAGENTA}{label}{CLR_RESET}")
    sys.stdout.flush()
    if pct >= 100.0:
        sys.stdout.write("\n")

# отрисовка цветной таблицы в трех цветах vokko
def _render_color_table(headers: List[str], rows: List[List[str]]) -> str:
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    widths = [w + 2 for w in widths]

    border_color = CLR_DIM
    sep_top = f"{border_color}+" + "+".join("-" * w for w in widths) + f"+{CLR_RESET}"
    sep_mid = f"{border_color}+" + "+".join("=" * w for w in widths) + f"+{CLR_RESET}"
    sep_bot = f"{border_color}+" + "+".join("-" * w for w in widths) + f"+{CLR_RESET}"

    lines = [sep_top]
    h_parts = []
    for i in range(cols):
        h_parts.append(f" {CLR_BOLD}{CLR_GREEN}{headers[i]:<{widths[i]-1}}{CLR_RESET}")
    lines.append(f"{border_color}|{CLR_RESET}" + f"{border_color}|{CLR_RESET}".join(h_parts) + f"{border_color}|{CLR_RESET}")
    lines.append(sep_mid)

    for r in rows:
        r_parts = []
        for i in range(cols):
            val = str(r[i])
            color = CLR_CYAN
            if i == 0:
                color = f"{CLR_BOLD}{CLR_MAGENTA}"
            elif val in ["[ЕСТЬ]", "[ГОТОВО]"]:
                color = f"{CLR_BOLD}{CLR_GREEN}"
            elif val == "[НЕТ]":
                color = CLR_DIM
            elif "RU" in val or "Русский" in val:
                color = f"{CLR_BOLD}{CLR_MAGENTA}"
            r_parts.append(f" {color}{val:<{widths[i]-1}}{CLR_RESET}")
        lines.append(f"{border_color}|{CLR_RESET}" + f"{border_color}|{CLR_RESET}".join(r_parts) + f"{border_color}|{CLR_RESET}")

    lines.append(sep_bot)
    return "\n".join(lines)

# запуск интерактивного мастера настройки с сохранением точки прогресса
def run_wizard_if_needed() -> bool:
    state = _load_wizard_state()

    # если мастер уже был пройден или явно пропущен пользователем
    if state.get("status") in ["completed", "skipped"]:
        return True

    print("\n" + f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")
    print(f"  {CLR_BOLD}{CLR_CYAN}VOKKO{CLR_RESET} {CLR_BOLD}{CLR_GREEN}//{CLR_RESET} {CLR_BOLD}{CLR_MAGENTA}МАСТЕР ПЕРВИЧНОЙ НАСТРОЙКИ НЕЙРОСЕТЕЙ И МОДЕЛЕЙ{CLR_RESET}")
    print(f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")

    while True:
        cur_step = state.get("step", "ask_start")

        # этап предложения пройти мастер
        if cur_step == "ask_start":
            print(f"\n{CLR_CYAN}Добро пожаловать в платформу Vokko{CLR_RESET}")
            print(f"{CLR_DIM}Вы можете пройти быструю настройку и выбрать модели для распознавания")
            print(f"и нейрокоррекции, либо пропустить и перейти сразу в веб-интерфейс{CLR_RESET}\n")

            try:
                ans = input(f"{CLR_CYAN}Желаете пройти настройку и выбрать модели? [{CLR_BOLD}{CLR_MAGENTA}Д{CLR_RESET}{CLR_CYAN}/н]: {CLR_RESET}").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{CLR_MAGENTA}Прервано пользователем, состояние сохранено{CLR_RESET}")
                return False

            if ans in ["н", "n", "no", "нет"]:
                state["status"] = "skipped"
                _save_wizard_state(state)
                print(f"{CLR_GREEN}Мастер настройки пропущен, запуск веб-сервера...{CLR_RESET}")
                return True

            state["status"] = "in_progress"
            state["step"] = "select_asr"
            _save_wizard_state(state)
            continue

        # этап выбора моделей распознавания речи
        asr_keys = list(WHISPER_MODELS.keys())
        if cur_step == "select_asr":
            print("\n" + f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")
            print(f"  {CLR_BOLD}{CLR_CYAN}ШАГ 1/3: ВЫБОР МОДЕЛЕЙ РАСПОЗНАВАНИЯ РЕЧИ (WHISPER ASR){CLR_RESET}")
            print(f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")

            asr_headers = ["№", "Модель (Параметры)", "Точность / WER", "Скорость", "VRAM", "SSD", "Наличие"]
            asr_rows = []
            for idx, k in enumerate(asr_keys, 1):
                m = WHISPER_MODELS[k]
                has = "[ЕСТЬ]" if is_model_downloaded(k) else "[НЕТ]"
                asr_rows.append([
                    str(idx),
                    m["name"],
                    m.get("accuracy", m["quality"]),
                    m["speed"],
                    m["vram"],
                    m["ssd"],
                    has
                ])

            print(_render_color_table(asr_headers, asr_rows))
            print(f"{CLR_DIM}Рекомендация: {CLR_BOLD}{CLR_MAGENTA}5 (Whisper Large v3 Turbo){CLR_RESET} {CLR_DIM}— оптимальный баланс скорости и точности{CLR_RESET}")
            print(f"{CLR_DIM}Для возврата назад введите: {CLR_BOLD}{CLR_CYAN}b / back / назад{CLR_RESET}")

            try:
                user_input = input(f"\n{CLR_CYAN}Выберите номера через запятую [{CLR_BOLD}{CLR_MAGENTA}по умолчанию: 5{CLR_RESET}{CLR_CYAN}]: {CLR_RESET}").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{CLR_MAGENTA}Прервано, прогресс сохранен на шаге выбора моделей Whisper{CLR_RESET}")
                return False

            # обработка возврата назад
            if user_input in ["b", "back", "назад"]:
                state["step"] = "ask_start"
                _save_wizard_state(state)
                continue

            selected_asr = []
            if not user_input:
                selected_asr = ["turbo"]
            else:
                parts = user_input.replace(" ", "").split(",")
                for p in parts:
                    if p.isdigit():
                        num = int(p)
                        if 1 <= num <= len(asr_keys):
                            selected_asr.append(asr_keys[num - 1])

            if not selected_asr:
                selected_asr = ["turbo"]

            state["selected_asr"] = selected_asr
            state["step"] = "select_llm"
            _save_wizard_state(state)
            print(f"{CLR_GREEN}Выбрано для Whisper: {', '.join(selected_asr)}{CLR_RESET}")
            continue

        # этап выбора языковых моделей для нейрокоррекции
        llm_keys = list(LLM_MODELS.keys())
        if cur_step == "select_llm":
            print("\n" + f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")
            print(f"  {CLR_BOLD}{CLR_CYAN}ШАГ 2/3: ВЫБОР МОДЕЛЕЙ НЕЙРОКОРРЕКЦИИ (LLM / SLM GGUF / ОБЛАКО){CLR_RESET}")
            print(f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")

            llm_headers = ["№", "Модель", "Категория", "Качество", "Скорость", "VRAM", "SSD", "Статус"]
            llm_rows = []
            for idx, k in enumerate(llm_keys):
                m = LLM_MODELS[k]
                has = "[ГОТОВО]" if (is_model_downloaded(k) or k in ["none", "custom-ollama", "openrouter"]) else "[НЕТ]"
                llm_rows.append([
                    str(idx),
                    m["name"],
                    m.get("category", ""),
                    m.get("quality", ""),
                    m["speed"],
                    m["vram"],
                    m["ssd"],
                    has
                ])

            print(_render_color_table(llm_headers, llm_rows))
            print(f"{CLR_DIM}0 - Без нейрокоррекции (только нативный вывод Whisper){CLR_RESET}")
            print(f"{CLR_DIM}4 - {CLR_BOLD}{CLR_MAGENTA}Llama 3.1 8B{CLR_RESET} {CLR_DIM}(рекомендуемый баланс под 8GB VRAM){CLR_RESET}")
            print(f"{CLR_DIM}12 - {CLR_BOLD}{CLR_MAGENTA}OpenRouter API{CLR_RESET} {CLR_DIM}(облачный доступ к любым флагманам){CLR_RESET}")
            print(f"{CLR_DIM}Для возврата назад введите: {CLR_BOLD}{CLR_CYAN}b / back / назад{CLR_RESET}")

            try:
                user_input = input(f"\n{CLR_CYAN}Выберите номера через запятую [{CLR_BOLD}{CLR_MAGENTA}по умолчанию: 4 (Llama 3.1 8B){CLR_RESET}{CLR_CYAN}]: {CLR_RESET}").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{CLR_MAGENTA}Прервано, прогресс сохранен на шаге выбора моделей LLM{CLR_RESET}")
                return False

            # обработка возврата назад
            if user_input in ["b", "back", "назад"]:
                state["step"] = "select_asr"
                _save_wizard_state(state)
                continue

            selected_llm = []
            if not user_input:
                selected_llm = ["llama3.1-8b"]
            else:
                parts = user_input.replace(" ", "").split(",")
                for p in parts:
                    if p.isdigit():
                        num = int(p)
                        if 0 <= num < len(llm_keys):
                            selected_llm.append(llm_keys[num])

            if not selected_llm:
                selected_llm = ["llama3.1-8b"]

            cfg = load_app_config()

            # дополнительная настройка openrouter если выбран
            if "openrouter" in selected_llm:
                print(f"\n{CLR_CYAN}Настройка OpenRouter API:{CLR_RESET}")
                cur_key = cfg.get("openrouter_api_key", "")
                key_prompt = f"Введите API-ключ OpenRouter [{CLR_BOLD}{CLR_MAGENTA}{'сохранен' if cur_key else 'нет'}{CLR_RESET}{CLR_CYAN}]: {CLR_RESET}"
                try:
                    entered_key = input(key_prompt).strip()
                    if entered_key:
                        cfg["openrouter_api_key"] = entered_key
                    cur_model = cfg.get("openrouter_model", "google/gemma-2-9b-it")
                    model_prompt = f"Введите модель OpenRouter [{CLR_BOLD}{CLR_MAGENTA}{cur_model}{CLR_RESET}{CLR_CYAN}]: {CLR_RESET}"
                    entered_model = input(model_prompt).strip()
                    if entered_model:
                        cfg["openrouter_model"] = entered_model
                    save_app_config(cfg)
                except (KeyboardInterrupt, EOFError):
                    return False

            # дополнительный опрос локальных моделей ollama если выбран
            if "custom-ollama" in selected_llm:
                ollama_models = _fetch_ollama_local_models()
                if ollama_models:
                    print(f"\n{CLR_CYAN}Доступные модели на локальном сервере Ollama:{CLR_RESET}")
                    for idx, om in enumerate(ollama_models):
                        print(f"  [{CLR_BOLD}{CLR_MAGENTA}{idx}{CLR_RESET}] {om}")
                    cur_om = cfg.get("ollama_model", ollama_models[0])
                    def_idx = 0
                    if cur_om in ollama_models:
                        def_idx = ollama_models.index(cur_om)
                    try:
                        om_choice = input(f"{CLR_CYAN}Выберите номер модели Ollama [{CLR_BOLD}{CLR_MAGENTA}по умолчанию: {def_idx} ({ollama_models[def_idx]}){CLR_RESET}{CLR_CYAN}]: {CLR_RESET}").strip()
                        if om_choice.isdigit() and int(om_choice) < len(ollama_models):
                            cfg["ollama_model"] = ollama_models[int(om_choice)]
                        else:
                            cfg["ollama_model"] = ollama_models[def_idx]
                        save_app_config(cfg)
                    except (KeyboardInterrupt, EOFError):
                        return False

            state["selected_llm"] = selected_llm
            state["step"] = "downloading"
            _save_wizard_state(state)
            print(f"{CLR_GREEN}Выбрано для нейрокоррекции: {', '.join(selected_llm)}{CLR_RESET}")
            continue

        # этап скачивания выбранных весов с индикатором
        if cur_step == "downloading":
            print("\n" + f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")
            print(f"  {CLR_BOLD}{CLR_CYAN}ШАГ 3/3: ЗАГРУЗКА ВЫБРАННЫХ МОДЕЛЕЙ{CLR_RESET}")
            print(f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")

            all_to_download = []
            for asr_id in state.get("selected_asr", []):
                if not is_model_downloaded(asr_id):
                    all_to_download.append((asr_id, WHISPER_MODELS[asr_id]["name"]))

            for llm_id in state.get("selected_llm", []):
                if llm_id not in ["none", "custom-ollama", "openrouter"] and not is_model_downloaded(llm_id):
                    all_to_download.append((llm_id, LLM_MODELS[llm_id]["name"]))

            if not all_to_download:
                print(f"{CLR_GREEN}Все выбранные модели уже готовы к работе{CLR_RESET}")
            else:
                print(f"{CLR_MAGENTA}Требуется загрузить моделей: {len(all_to_download)}{CLR_RESET}\n")
                for m_id, m_name in all_to_download:
                    print(f"{CLR_CYAN}Загрузка: {m_name}...{CLR_RESET}")
                    cb = lambda d, t, p, name=m_name: _cli_progress_bar(d, t, p, name)
                    success = download_model_file(m_id, progress_callback=cb)
                    if not success:
                        print(f"{CLR_MAGENTA}Не удалось загрузить {m_name}, вы сможете докачать ее позже из интерфейса{CLR_RESET}")

            # сохранение активных моделей в конфигурацию
            cfg = load_app_config()
            if state.get("selected_asr"):
                cfg["active_asr_model"] = state["selected_asr"][-1]
            if state.get("selected_llm"):
                cfg["active_llm_model"] = state["selected_llm"][-1]
            cfg["wizard_completed"] = True
            save_app_config(cfg)

            state["status"] = "completed"
            _save_wizard_state(state)

            print("\n" + f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}")
            print(f"  {CLR_BOLD}{CLR_GREEN}НАСТРОЙКА УСПЕШНО ЗАВЕРШЕНА! ЗАПУСК ВЕБ-СЕРВЕРА...{CLR_RESET}")
            print(f"{CLR_DIM}" + "=" * 80 + f"{CLR_RESET}\n")
            time.sleep(1.0)
            return True

    return True
