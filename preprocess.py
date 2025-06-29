import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm
import pymorphy3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize morphological analyzer and stopwords once
morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))
punctuation_marks = set(['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--'])

def preprocess(text, stop_words, punctuation_marks, morph):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return ' '.join(preprocessed_text)  # Join tokens back to string for vectorizer

# Load dataset
news = pd.read_csv('lenta-ru-news.csv', low_memory=False)

# Select topics and sample
topics = ['Путешествия', 'Ценности', 'Мир', 'Наука и техника', 'Экономика']
news_in_cat_count = 2000

df_res = pd.DataFrame()
for topic in tqdm(topics):
    df_topic = news[news['topic'] == topic].head(news_in_cat_count)
    df_res = pd.concat([df_res, df_topic], ignore_index=True)

df_res['Preprocessed_texts'] = df_res['text'].apply(
    lambda x: preprocess(x, stop_words, punctuation_marks, morph)
)

X = df_res['Preprocessed_texts']
y = df_res['topic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
my_tags = df_res['topic'].unique()

nb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=my_tags))

model_path = 'lenta_nb_model.joblib'
joblib.dump(nb, model_path)  # Fixed variable name from pipeline to nb
print(f"Model saved to {model_path}")
