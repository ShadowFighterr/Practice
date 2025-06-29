import os
import joblib
import string
import psycopg2
from datetime import datetime

from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer

# --- Настройка NLTK (для скачивания необходимых данных) ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt_tab')
except nltk.downloader.DownloadError:
    nltk.download('punkt_tab')

# --- Функция предобработки текста ---
def preprocess(text, stop_words, punctuation_marks, morph):
    """
    Выполняет предобработку текста.
    """
    if not isinstance(text, str):
        return ""

    tokens = word_tokenize(text.lower())
    cleaned_tokens = [
        morph.parse(token)[0].normal_form
        for token in tokens
        if token not in stop_words and token not in punctuation_marks and token.isalnum()
    ]
    return " ".join(cleaned_tokens)

# --- Инициализация глобальных переменных для предобработки ---
stop_words = set(stopwords.words('russian'))
punctuation_marks = set(string.punctuation)
morph = MorphAnalyzer()

# --- Конфигурация базы данных PostgreSQL ---
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Azamat06") # Вставьте ваш пароль здесь!
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# --- Загрузка обученной модели ---
model_path = 'lenta_nb_model.joblib'
try:
    model = joblib.load(model_path)
    print("Модель классификации успешно загружена.")
except FileNotFoundError:
    print(f"Ошибка: Файл модели '{model_path}' не найден.")
    exit(1)
except Exception as e:
    print(f"Произошла ошибка при загрузке модели: {e}")
    exit(1)

# --- Инициализация FastAPI ---
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# --- Зависимость для получения соединения с БД ---
def get_db():
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        yield conn
    except Exception as e:
        print(f"Ошибка подключения к базе данных: {e}")
    finally:
        if conn:
            conn.close()

# --- Вспомогательная функция для получения истории предсказаний ---
def get_predictions_history(db_conn, limit=10):
    history = []
    if db_conn:
        try:
            cur = db_conn.cursor()
            cur.execute("SELECT news_text, prediction, timestamp FROM predictions_history ORDER BY timestamp DESC LIMIT %s;", (limit,))
            history_records = cur.fetchall()
            cur.close()
            for record in history_records:
                text, pred, ts = record
                history.append({
                    "text": text,
                    "prediction": pred,
                    "timestamp": ts.strftime('%Y-%m-%d %H:%M:%S'),
                    "short_text": text[:50] + "..." if len(text) > 50 else text
                })
        except Exception as e:
            print(f"Ошибка при загрузке истории предсказаний из базы данных: {e}")
    return history

# --- Маршруты FastAPI ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db_conn: psycopg2.extensions.connection = Depends(get_db)):
    """
    Отображает главную страницу с формой и историей предсказаний.
    """
    history = get_predictions_history(db_conn)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "history": history})

@app.post("/", response_class=HTMLResponse)
async def classify_news(request: Request, news_text: str = Form(...), db_conn: psycopg2.extensions.connection = Depends(get_db)):
    """
    Обрабатывает отправку формы, делает предсказание и сохраняет его в БД,
    предварительно проверяя на дубликаты.
    """
    prediction_result = None
    message_type = "success" # Для отображения типа сообщения в шаблоне

    if news_text:
        preprocessed_text = preprocess(news_text, stop_words, punctuation_marks, morph)
        try:
            prediction = model.predict([preprocessed_text])[0]
            prediction_result = prediction.upper()

            # --- Сохранение предсказания в БД ---
            if db_conn:
                try:
                    cur = db_conn.cursor()

                    # Проверяем, существует ли уже такая запись
                    check_query = """
                    SELECT COUNT(*) FROM predictions_history
                    WHERE news_text = %s AND prediction = %s;
                    """
                    cur.execute(check_query, (news_text, prediction_result))
                    count = cur.fetchone()[0]

                    if count > 0:
                        print(f"Дубликат обнаружен. Запись с текстом '{news_text[:50]}...' и предсказанием '{prediction_result}' уже существует.")
                        prediction_result = f"Предсказанная категория: {prediction_result}. Запись уже существует в истории."
                        message_type = "info" # Используем info для сообщения о дубликате
                    else:
                        insert_query = """
                        INSERT INTO predictions_history (news_text, prediction, timestamp)
                        VALUES (%s, %s, %s);
                        """
                        cur.execute(insert_query, (news_text, prediction_result, datetime.now()))
                        db_conn.commit()
                        print("Предсказание успешно сохранено в базе данных.")
                        prediction_result = f"Предсказанная категория: {prediction_result}" # Возвращаем чистое предсказание для success
                        message_type = "success" # Успешное сохранение

                    cur.close()

                except Exception as e:
                    print(f"Ошибка при сохранении/проверке предсказания в базе данных: {e}")
                    db_conn.rollback()
                    prediction_result = "Ошибка при сохранении в БД."
                    message_type = "warning"
            else:
                prediction_result = "Не удалось сохранить предсказание: нет соединения с базой данных."
                message_type = "warning"

        except Exception as e:
            print(f"Произошла ошибка при предсказании: {e}")
            prediction_result = "Ошибка предсказания."
            message_type = "warning"
    else:
        prediction_result = "Пожалуйста, введите текст новости."
        message_type = "warning"

    # После обработки запроса, получаем обновленную историю
    history = get_predictions_history(db_conn)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction_result,
        "history": history,
        "user_input": news_text,
        "message_type": message_type # Передаем тип сообщения в шаблон
    })

# Новый эндпоинт для получения истории предсказаний (полезно для AJAX)
@app.get("/history", response_class=HTMLResponse)
async def get_history_html(request: Request, db_conn: psycopg2.extensions.connection = Depends(get_db)):
    history = get_predictions_history(db_conn)
    return templates.TemplateResponse("history_sidebar.html", {"request": request, "history": history})
