from flask import Flask, request, jsonify, render_template
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string

# Download NLTK resources
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# Load model and vectorizer
pac = joblib.load('passiveaggressive_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        article_text = data.get('article_text', '')
        
        if not article_text.strip():
            return jsonify({'error': 'Please enter some text to analyze'}), 400
        
        preprocessed_text = preprocess_text(article_text)
        tfidf_text = tfidf_vectorizer.transform([preprocessed_text])
        prediction = pac.predict(tfidf_text)[0]
        
        # Get decision function score
        decision_score = pac.decision_function(tfidf_text)[0]
        
        # Calculate confidence using sigmoid function to normalize the decision score
        # This gives a more meaningful probability-like value
        import numpy as np
        confidence = 1 / (1 + np.exp(-abs(decision_score)))
        
        return jsonify({
            'prediction': prediction, 
            'confidence': float(confidence),
            'decision_score': float(decision_score)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
