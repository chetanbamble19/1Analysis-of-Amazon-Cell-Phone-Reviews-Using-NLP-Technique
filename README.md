# Analysis of Amazon Cell Phone Reviews Using NLP Technique

This project is a Flask application designed to perform sentiment analysis on Amazon cell phone reviews using NLP techniques. It allows users to input reviews, then analyzes and classifies the sentiment (positive or negative) using a trained machine learning model.

## Project Structure

- **/templates/**: Contains HTML templates for user input forms and sentiment analysis result display.
- **/model/**: Directory for storing the trained sentiment analysis model and tokenizer.
- **app.py**: Main Flask application file that handles routes, prediction logic, and user interactions.
- **train.py**: Script for training the sentiment analysis model. It preprocesses the data, builds, trains, and saves the model.
- **requirements.txt**: List of required libraries and dependencies for the project.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (if needed)**  
   To train the sentiment analysis model on your own dataset, run:

   ```bash
   python train.py
   ```

   This will generate model files within the `/model` directory.

4. **Run the Application**  
   Start the Flask application with:
   ```bash
   python app.py
   ```

## Usage

1. **Input**: Users can enter an Amazon review directly into the application.
2. **Prediction**: Click the "Predict" button to analyze the sentiment of the entered review.
3. **Result**: The application displays whether the sentiment is positive or negative.

## Key Features

- **Interactive Sentiment Analysis**: Simple and intuitive user interface for analyzing review sentiments.
- **NLP-Powered Model**: Model trained on a comprehensive dataset of Amazon cell phone reviews for accurate predictions.
- **Expandable Architecture**: Easily retrain or fine-tune the model with new datasets as needed.

## Requirements

The project requires Python 3.x and the dependencies listed in `requirements.txt`. Key libraries include Flask for the web framework, TensorFlow or PyTorch for NLP modeling, and NLTK or SpaCy for natural language processing.

## Future Enhancements

- Add more sentiment categories (e.g., neutral).
- Improve model accuracy with hyperparameter tuning and additional datasets.
- Extend the user interface with visualized sentiment breakdowns.
