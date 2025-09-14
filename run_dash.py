from app import app  # импортируем объект Dash
from werkzeug.serving import run_simple

if __name__ == "__main__":
    # запускаем сервер на локальном интерфейсе
    run_simple('127.0.0.1', 8050, app.server, use_reloader=True, use_debugger=True)