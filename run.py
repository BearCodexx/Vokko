import backend.app.core.doh
import backend.app.core.silence
import webbrowser
import threading
import time
import uvicorn
from backend.app.core.config import HOST, PORT
from backend.app.core.logger import print_banner, log_info
from backend.app.core.wizard import run_wizard_if_needed

# открытие адреса в браузере
def open_browser():
    time.sleep(1.2)
    webbrowser.open(f"http://{HOST}:{PORT}")

# запуск веб сервера
if __name__ == "__main__":
    print_banner()
    log_info("Инициализация платформы Vokko")

    # запуск мастера
    wizard_ok = run_wizard_if_needed()

    log_info(f"Сервер запущен и доступен по адресу http://{HOST}:{PORT}")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.app.main:app", host=HOST, port=PORT, reload=False, log_level="critical", timeout_keep_alive=300)

