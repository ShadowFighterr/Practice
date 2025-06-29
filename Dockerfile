FROM python:3.12-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

COPY . .

RUN if [ ! -f "lenta_nb_model.joblib" ]; then echo "Error: lenta_nb_model.joblib not found!"; exit 1; fi

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
