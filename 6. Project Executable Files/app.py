from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
model = load_model('model/sentiment_model.h5')

# Load the tokenizer
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    
    # Preprocess the review text
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed

    # Make a prediction
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return render_template('result.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
