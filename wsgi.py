from app import app  # импортируем твой Dash app

server = app.server  # Gunicorn будет использовать Flask сервер