import jwt  # убедись, что установлен pyjwt: pip install pyjwt

SECRET_KEY = "super-secret-key"

# Генерация токена
token = jwt.encode({"user": "admin"}, SECRET_KEY, algorithm="HS256")
print(token)

# Декодирование токена
payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
print(payload)