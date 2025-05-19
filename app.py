from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
# Load trained pipeline (preprocess_texts is imported via text_utils)
model = joblib.load('lenta_nb_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form.get('news_text')
        if text:
            prediction = model.predict([text])[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
