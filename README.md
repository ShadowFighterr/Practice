Ubaidullauly Azamat. IT-2305. Industrial Practice.

```markdown
# News Classifier

This is an interactive web application for classifying news articles into categories using a pre-trained machine learning model. The application is developed with FastAPI for the backend, PostgreSQL for storing prediction history, Jinja2 for rendering HTML templates, and CSS and JavaScript for a dynamic user interface.

## Description

The application allows the user to enter news text, get a predicted category based on a trained Naive Bayes model, and saves each prediction to a database. Previous predictions are displayed in a convenient sidebar with expandable sections for viewing the full news text and result.

## Key Features

- **Text Classification:** Uses a trained model to determine the news category.
- **Web Interface:** Intuitive interface built on FastAPI, Jinja2, HTML, CSS, and JavaScript.
- **Prediction History:** Stores all predictions in a PostgreSQL database.
- **UI testing:** Streamlit framework was used as a tool to test the project for detailed viewing.

## Project Structure

```

.
├── app.py                      # Main FastAPI application file, classification logic, and DB interaction.
├── lenta\_nb\_model.joblib       # Pre-trained Naive Bayes model file (loaded by the application).
├── preprocess.py               # (Original) text preprocessing logic, whose functions are integrated into app.py.
├── requirements.txt            # List of Python dependencies.
├── static/                     # Directory for static files (CSS, JavaScript).
│   └── style.css               # Stylesheet for UI design.
├── templates/                  # Directory for HTML-templates Jinja2.
│   └── index.html              # Main HTML template for the web page.
└── venv/                       # Python virtual environment (recommended).

````

## Installation and Setup

To run the application, follow these steps:

### 1. Prerequisites

**Python 3.x:** Ensure you have Python version 3.7 or higher installed.

**PostgreSQL:** Install and configure PostgreSQL on your system.

PostgreSQL Installation (Ubuntu):

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
````

Setting password for postgres user:

```bash
sudo -i -u postgres
psql
\password postgres
# Enter and confirm your password (e.g., Azamat06)
\q
exit
```

**pgAdmin 4 (recommended):** A graphical tool for managing PostgreSQL.

pgAdmin 4 Installation (Ubuntu):

```bash
curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg
sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'
sudo apt install pgadmin4
```

### 2. Database Setup

Using pgAdmin 4 or psql, create the database and table for storing predictions:

Create a database: `news_predictions_db` (or use `postgres`).

Create the `predictions_history` table:

```sql
CREATE TABLE predictions_history (
    id SERIAL PRIMARY KEY,
    news_text TEXT NOT NULL,
    prediction VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);
```

### 3. Python Project Setup

Clone the repository (if applicable) or ensure you have all project files.

Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(Ensure `requirements.txt` contains `fastapi`, `uvicorn[standard]`, `jinja2`, `psycopg2-binary`, `nltk`, `pymorphy3`, `joblib`, `scikit-learn` - if the model uses it).

If `requirements.txt` is missing, you can create it by running:

```bash
pip install fastapi uvicorn[standard] jinja2 psycopg2-binary nltk pymorphy3 joblib scikit-learn
pip freeze > requirements.txt
```

Download necessary NLTK data (this is done automatically in your `app.py`, but it's good to ensure):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
# Possibly also 'punkt_tab' if required for pymorphy
```

Place the model: Ensure `lenta_nb_model.joblib` is in the project's root directory, next to `app.py`.

Update DB Configuration: Open `app.py` and MUST replace `your_strong_password` with your actual password for the `postgres` user in the `DB_PASSWORD` line:

```python
DB_PASSWORD = os.getenv("DB_PASSWORD", "Azamat06") # <-- Ensure this is your password
```

### 4. Running the Application

Start the FastAPI server from the project's root directory:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The application will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Usage

Open the specified address in your web browser.

In the center of the page, you will see a field for entering news text and a "Predict Category" button.

On the left, there will be a sidebar with "Prediction History".

Enter news text into the input field and click "Predict Category".

The application will display the predicted category and automatically save the entry to history.

If you try to save the same news with the same prediction, the application will inform you that the entry already exists and will not create a duplicate.

Each history entry in the sidebar is presented as a brief title. Click on the title to expand and see the full news text and its predicted category.

## Functionality

* **Classification:** Input field for news text, which is processed and classified.
* **Result Display:** The predicted category is displayed on the main page.
* **DB Storage:** The entered text, predicted category, and timestamp are saved to the `predictions_history` table in PostgreSQL.
* **Duplicate Check:** A check is performed when submitting a new prediction to avoid re-saving identical data.
* **Sidebar History:** A list of recent predictions is displayed in an interactive sidebar.
* **Expandable Elements:** Each history entry is presented as an "accordion," allowing you to hide or show the full news text and prediction result.

## Technologies

* **Python:** Main development language.
* **FastAPI:** High-performance web framework for building APIs.
* **Uvicorn:** ASGI server for running FastAPI.
* **Jinja2:** Templating engine for generating dynamic HTML.
* **PostgreSQL:** Relational database for persistent data storage.
* **Psycopg2:** PostgreSQL adapter for Python.
* **NLTK (Natural Language Toolkit):** Library for natural language processing (tokenization, stop words).
* **PyMorphy3:** Morphological analyzer for the Russian language (lemmatization).
* **Joblib:** For serializing/deserializing trained models.
* **Scikit-learn:** Assumed that the `lenta_nb_model.joblib` model was trained using this library.
* **HTML, CSS, JavaScript:** For frontend development and interactivity.
* **Streamlit:** For analyzing predictions, collecting logs, and testing functionality.

## Deployment

I have deployed the project on Render.com using Docker and also stored a database there: https://practice-fkig.onrender.com/
