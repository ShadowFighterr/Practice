Machine Learning Algorithms final project.
Group: IT-2305.
Team members: Azamat Ubaidullauly, Astana Khunakhan
Instructor: Ainur Mukashova

Deployed website: https://mlprediction-model.onrender.com/

```markdown
# Lenta.ru News Topic Classification

This repository contains a machine learning pipeline and a simple web application for classifying Lenta.ru news articles into predefined topics. The pipeline uses text preprocessing in Russian, vectorization, and a Multinomial Naive Bayes classifier. A Flask-based web interface allows users to input arbitrary news text and get a predicted topic.

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Comparative Analysis of Methods](#comparative-analysis-of-methods)  
- [Why Multinomial Naive Bayes?](#why-multinomial-naive-bayes)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Data Preprocessing & Model Training](#data-preprocessing--model-training)  
  - [Launching the Web App](#launching-the-web-app)  
- [Dependencies](#dependencies)  
- [Contributing](#contributing)  
- [Contact](#contact)  

---

## Project Overview

This project aims to build a multiclass text classifier that predicts the topic of a Russian-language news article from the Lenta.ru dataset. We select five topics:  
- **Путешествия** (Travel)  
- **Ценности** (Values)  
- **Мир** (World)  
- **Наука и техника** (Science & Technology)  
- **Экономика** (Economics)  

After preprocessing and vectorizing the text, various classification algorithms were compared. The final choice—Multinomial Naive Bayes—balances high accuracy with low inference/training time.

A simple Flask web application (`app.py`) allows users to enter arbitrary news text and receive a predicted topic label in real time.

---

## Dataset

We use the publicly available Lenta.ru news dataset (`lenta-ru-news.csv`). For each of the five selected topics, we sample up to 2,000 articles, yielding a combined subset for training and evaluation.

- **Dataset source**: [Lenta.ru News CSV](https://github.com/your-repo/MLfinal/blob/main/lenta-ru-news.csv)  
- **Topics included**:
  - Путешествия
  - Ценности
  - Мир
  - Наука и техника
  - Экономика  

---

## Comparative Analysis of Methods

Below is a summary of classification algorithms we evaluated on the same training/test split (70% train, 30% test):

| Method                             | Training/Inference Time | Accuracy            |
| ---------------------------------- | ----------------------- | -------------------- |
| Multinomial Naive Bayes            | 1 min 22 sec            | 0.9163              |
| Gaussian Naive Bayes               | 13.1 sec                | 0.8357              |
| Multinomial Logistic Regression    | 12.4 sec                | 0.9350              |
| Decision Tree                      | 2.36 sec                | 0.6767              |
| Random Forest                      | 4.48 sec                | 0.8923              |
| XGBoost                            | 6 min 31 sec            | 0.9153              |
| LightGBM                           | 1 min 41 sec            | 0.9180              |
| Support Vector Machine (SVM)       | 2 min 55 sec            | 0.9477              |

---

## Why Multinomial Naive Bayes?

While SVM (Accuracy 0.9477) and Multinomial Logistic Regression (Accuracy 0.9350) achieved slightly higher accuracy, Multinomial Naive Bayes was chosen because:

1. **Speed**: It trains in approximately **1 min 22 sec** and makes predictions almost instantly, which is critical for rapid iteration and deploying a lightweight web service.  
2. **Competitive Accuracy**: With an accuracy of **0.9163**, it remains highly competitive compared to heavier algorithms.  
3. **Simplicity & Interpretability**: The model pipeline uses a straightforward count vectorizer → TF-IDF transformer → Naive Bayes classifier, making it easy to understand and maintain.

---

## Project Structure

```

MLfinal/
├── app.py                     # Flask web application
├── lenta\_nb\_model.joblib       # Trained MultinomialNB pipeline (auto-generated)
├── lenta-ru-news.csv           # Original Lenta.ru dataset (subset used)
├── preprocess.py               # Data preprocessing & model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── templates/
│   └── index.html             # HTML template for the web interface
├── **pycache**/
│   └── text\_utils.cpython-313.pyc
└── venv/                       # Python virtual environment (ignored by Git)
├── bin/
├── include/
├── lib/
├── lib64 -> lib
└── pyvenv.cfg

````

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ShadowFighterr/MLfinal.git
   cd MLfinal
````

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: If you run into issues installing `pymorphy3`, ensure that `pymorphy3-dicts-ru` is installed. You can do:
   >
   > ```bash
   > pip install pymorphy3 pymorphy3-dicts-ru
   > ```

---

## Usage

### Data Preprocessing & Model Training

1. **Run `preprocess.py`**
   This script will:

   * Load and filter the Lenta.ru dataset to the five chosen topics (2,000 articles each).
   * Perform Russian-language text preprocessing (tokenization, lemmatization via `pymorphy3`, removal of stop words & punctuation).
   * Vectorize the preprocessed text with `CountVectorizer` → `TfidfTransformer`.
   * Train a `MultinomialNB` pipeline and evaluate accuracy on a held-out test set.
   * Save the trained model to `lenta_nb_model.joblib`.

   ```bash
   python preprocess.py
   ```

2. **Verify the model**
   After successful training, you should see printed output similar to:

   ```
   accuracy 0.9163333333333333
   (classification report with precision, recall, f1-score for each topic)
   Model saved to lenta_nb_model.joblib
   ```

   At this point, `lenta_nb_model.joblib` will exist in the project root.

### Launching the Web App

1. **Ensure the model file is present**
   Confirm that `lenta_nb_model.joblib` is in the project directory (if you did not train locally, you can download it from the GitHub release or from a teammate).

2. **Run the Flask app**

   ```bash
   python app.py
   ```

   By default, Flask runs on `http://127.0.0.1:5000/` in debug mode.

3. **Open the interface**
   Navigate to `http://127.0.0.1:5000/` in your web browser. You will see a simple form prompting you to paste or type in news text. After clicking **Классифицировать**, the predicted topic (one of the five) will be displayed.

---

## Dependencies

All required packages are listed in `requirements.txt`. Key libraries include:

* **pandas**, **numpy**, **scikit-learn**, **joblib**
  For data handling, model building, and serialization.
* **nltk**, **pymorphy3**, **pymorphy3-dicts-ru**
  For Russian-language tokenization, lemmatization, and stopword removal.
* **Flask** (or optionally FastAPI/uvicorn)
  For serving the web application.
* **tqdm**
  To show progress bars during dataset filtering.
* **pydantic**, **uvicorn**, **fastapi**
  Included in case you prefer to swap out Flask for FastAPI in the future.

---

## Contributing

1. Fork this repository.
2. Create a new feature branch:

   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. Make your changes (e.g., add new preprocessing steps, test different classifiers, improve the UI).
4. Commit and push:

   ```bash
   git commit -m "Add something useful"
   git push origin feature/YourFeatureName
   ```
5. Open a pull request.

Please ensure that any new code is documented, dependencies are updated in `requirements.txt`, and your changes do not break the existing pipeline.

---

## Contact

If you have questions or suggestions, feel free to open an issue or reach out to the repository owner.
GitHub: [ShadowFighterr/MLfinal](https://github.com/ShadowFighterr/MLfinal)

```

---
