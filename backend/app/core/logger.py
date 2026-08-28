import sys
import os
from datetime import datetime

# перенастройка кодировки стандартных потоков вывода на utf8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# цветовые коды для разнообразия вывода в терминале
CLR_RESET = "\033[0m"
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_BLUE = "\033[94m"
CLR_MAGENTA = "\033[95m"
CLR_RED = "\033[91m"
CLR_DIM = "\033[90m"
CLR_BOLD = "\033[1m"

# включение поддержки цветов в терминале windows
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

# вывод стилизованного заголовка приложения
def print_banner():
    banner_lines = [
        f"{CLR_CYAN}    ██╗   ██╗ ██████╗ ██╗  ██╗██╗  ██╗ ██████╗ {CLR_RESET}",
        f"{CLR_CYAN}    ██║   ██║██╔═══██╗██║ ██╔╝██║ ██╔╝██╔═══██╗{CLR_RESET}",
        f"{CLR_GREEN}    ██║   ██║██║   ██║█████╔╝ █████╔╝ ██║   ██║{CLR_RESET}",
        f"{CLR_GREEN}    ╚██╗ ██╔╝██║   ██║██╔═██╗ ██╔═██╗ ██║   ██║{CLR_RESET}",
        f"{CLR_MAGENTA}     ╚████╔╝ ╚██████╔╝██║  ██╗██║  ██╗╚██████╔╝{CLR_RESET}",
        f"{CLR_MAGENTA}      ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ {CLR_RESET}",
        f"{CLR_DIM}  ======================================================{CLR_RESET}",
        f"  {CLR_BOLD}{CLR_CYAN}VOKKO // DEEP SPEECH & MUSIC TRANSCRIPTION PLATFORM{CLR_RESET}",
        f"{CLR_DIM}  ======================================================{CLR_RESET}\n"
    ]
    try:
        sys.stdout.write("\n".join(banner_lines) + "\n")
        sys.stdout.flush()
    except Exception:
        pass

# вывод информационных сообщений в терминал с подсветкой
def log_info(message: str):
    time_str = datetime.now().strftime("%H:%M:%S")
    try:
        sys.stdout.write(f"{CLR_DIM}[{time_str}]{CLR_RESET} {CLR_CYAN}[ИНФО]{CLR_RESET} {CLR_GREEN}{message}{CLR_RESET}\n")
        sys.stdout.flush()
    except Exception:
        pass

# вывод этапов работы с выделением цветом
def log_stage(stage_num: int, total_stages: int, message: str):
    time_str = datetime.now().strftime("%H:%M:%S")
    try:
        sys.stdout.write(f"{CLR_DIM}[{time_str}]{CLR_RESET} {CLR_MAGENTA}[ЭТАП {stage_num}/{total_stages}]{CLR_RESET} {CLR_CYAN}{message}{CLR_RESET}\n")
        sys.stdout.flush()
    except Exception:
        pass

# вывод прогресса скачивания модели в реальном времени
def log_progress(current: int, total: int, label: str = "Скачивание модели"):
    pct = int((current / total) * 100) if total else 0
    filled = int(pct / 4)
    bar = "=" * filled + ">" if filled < 25 else "=" * 25
    bar_str = f"[{bar:<25}]"
    time_str = datetime.now().strftime("%H:%M:%S")
    mb_cur = current / (1024 * 1024)
    mb_tot = total / (1024 * 1024)
    line = f"\r{CLR_DIM}[{time_str}]{CLR_RESET} {CLR_MAGENTA}[ЗАГРУЗКА]{CLR_RESET} {CLR_CYAN}{label}:{CLR_RESET} {CLR_GREEN}{bar_str} {pct}% ({mb_cur:.1f} МБ / {mb_tot:.1f} МБ){CLR_RESET}"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    except Exception:
        pass

# вывод предупреждений в желтом цвете
def log_warning(message: str):
    time_str = datetime.now().strftime("%H:%M:%S")
    try:
        sys.stdout.write(f"{CLR_DIM}[{time_str}]{CLR_RESET} {CLR_YELLOW}[ВНИМАНИЕ]{CLR_RESET} {CLR_YELLOW}{message}{CLR_RESET}\n")
        sys.stdout.flush()
    except Exception:
        pass

# вывод сообщений об ошибках в красном цвете
def log_error(message: str):
    time_str = datetime.now().strftime("%H:%M:%S")
    try:
        sys.stderr.write(f"{CLR_DIM}[{time_str}]{CLR_RESET} {CLR_RED}[ОШИБКА]{CLR_RESET} {CLR_YELLOW}{message}{CLR_RESET}\n")
        sys.stderr.flush()
    except Exception:
        pass
