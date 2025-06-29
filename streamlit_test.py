import streamlit as st
import joblib
import nltk
import os
import string
import psycopg2
from datetime import datetime
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
DB_PASSWORD = os.getenv("DB_PASSWORD", "Azamat06") # <-- Вставьте ваш пароль здесь!
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# --- Функция для подключения к БД ---
@st.cache_resource
def get_db_connection():
    """
    Устанавливает и кэширует соединение с базой данных PostgreSQL.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        # st.success("Успешно подключено к базе данных PostgreSQL!") # Можно убрать для более чистого UI
        return conn
    except Exception as e:
        st.error(f"Ошибка подключения к базе данных: {e}")
        st.stop()
        return None

# --- Загрузка обученной модели ---
model_path = 'lenta_nb_model.joblib'
try:
    model = joblib.load(model_path)
    # st.success("Модель классификации успешно загружена!") # Можно убрать для более чистого UI
except FileNotFoundError:
    st.error(f"Ошибка: Файл модели '{model_path}' не найден. Убедитесь, что он находится в том же каталоге.")
    st.stop()
except Exception as e:
    st.error(f"Произошла ошибка при загрузке модели: {e}")
    st.stop()

# Получаем соединение с БД
conn = get_db_connection()

# --- Streamlit UI: Боковая панель ---
with st.sidebar:
    st.header("История предсказаний")
    if conn:
        try:
            cur = conn.cursor()
            # Выбираем последние 15 предсказаний для боковой панели
            cur.execute("SELECT news_text, prediction, timestamp FROM predictions_history ORDER BY timestamp DESC LIMIT 15;")
            history_records = cur.fetchall()
            cur.close()

            if history_records:
                for i, record in enumerate(history_records):
                    text, pred, ts = record
                    # Создаем "спойлер" для каждой записи
                    # Заголовок спойлера: первые 50 символов текста и предсказание
                    spoiler_title = f"({ts.strftime('%H:%M')}) {text[:50]}... ({pred})"
                    with st.expander(spoiler_title):
                        st.write(f"**Дата/Время:** {ts.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Исходный текст:**")
                        st.info(text) # Полный текст новости
                        st.write(f"**Предсказанная категория:**")
                        st.success(pred) # Предсказанная категория
                        st.markdown("---") # Разделитель внутри спойлера
            else:
                st.info("История предсказаний пока пуста. Сделайте первое предсказание!")

        except Exception as e:
            st.error(f"Ошибка при загрузке истории предсказаний из базы данных: {e}")
    else:
        st.warning("Не удалось загрузить историю предсказаний: нет соединения с базой данных.")
    st.markdown("---")
    st.info("Приложение разработано для классификации новостей.")


# --- Streamlit UI: Основная часть страницы ---
st.set_page_config(page_title="Классификатор новостей", layout="centered")

st.title("Классификатор новостей")
st.markdown("---")

st.header("Введите текст новости для классификации:")

user_input = st.text_area(
    "Текст новости:",
    "Например: В Астане открылся новый технологический парк, привлекающий молодых специалистов в сфере IT.",
    key="news_input"
)

if st.button("Предсказать категорию"):
    if user_input:
        preprocessed_text = preprocess(user_input, stop_words, punctuation_marks, morph)
        try:
            prediction = model.predict([preprocessed_text])[0]
            st.success(f"**Предсказанная категория:** {prediction.upper()}")

            # --- Сохранение предсказания в БД ---
            if conn:
                try:
                    cur = conn.cursor()
                    insert_query = """
                    INSERT INTO predictions_history (news_text, prediction, timestamp)
                    VALUES (%s, %s, %s);
                    """
                    cur.execute(insert_query, (user_input, prediction.upper(), datetime.now()))
                    conn.commit()
                    cur.close()
                    st.success("Предсказание успешно сохранено в базе данных! Обновите страницу или переоткройте приложение, чтобы увидеть в истории.")
                    # Чтобы история в сайдбаре обновилась, Streamlit должен перерандерить весь скрипт.
                    # Это произойдет автоматически при следующем взаимодействии пользователя
                    # или принудительно, если добавить st.rerun() (но это не всегда желательно).
                except Exception as e:
                    st.error(f"Ошибка при сохранении предсказания в базе данных: {e}")
                    conn.rollback()
            else:
                st.warning("Не удалось сохранить предсказание: нет соединения с базой данных.")

        except Exception as e:
            st.error(f"Произошла ошибка при предсказании: {e}")
    else:
        st.warning("Пожалуйста, введите текст новости перед нажатием кнопки 'Предсказать'.")

st.markdown("---")
st.info("Это приложение использует предварительно обученную модель для классификации новостей.")
