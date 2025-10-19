# wsgi.py
from app import main_app
from flask import session, request, redirect
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

SECRET_KEY = os.getenv("SALES_DASHBOARD_KEY")
FLASK_SESSION_SECRET = os.getenv("FLASK_SESSION_SECRET")



app = main_app()
app.server.secret_key = FLASK_SESSION_SECRET  


app.server.config.update(
    JSONIFY_TIMEOUT=30,
    SEND_FILE_MAX_AGE_DEFAULT=3600,
    PERMANENT_SESSION_LIFETIME=1800,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False,  # True если используете HTTPS
    SESSION_REFRESH_EACH_REQUEST=True
)


@app.server.before_request
def check_key():
    protected_pages = ('/', '/Sales_dimamix', '/Segments', '/Matrix')
    
    if request.path not in protected_pages:
        return

    #1. Сначала проверяем сессию
    if session.get('authenticated'):
        return  # Уже авторизован
    
    # 2. Если нет сессии - проверяем ключ
    key = request.args.get("key")
    if key == SECRET_KEY:  # Правильный ключ
        session['authenticated'] = True
        if 'key' in request.args:
            return redirect(request.path)
        return
    
    # 3. Неправильный ключ и нет сессии
    return "Unauthorized", 401

server = app.server

if __name__ == "__main__":
    app.run(
        debug=False, 
        host='0.0.0.0', 
        port=8050,
        threaded=True,      # Многопоточность для параллельных запросов
        processes=1         # Один процесс для стабильности
    )