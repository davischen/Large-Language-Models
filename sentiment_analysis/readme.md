# Sentiment Analysis System

## Overview
This project implements a **Sentiment Analysis System** that classifies text as having either a positive or negative sentiment. The system uses various feature extraction methods and machine learning models, including Perceptron and Logistic Regression classifiers, to learn and predict sentiment based on input text.

## Features

### 1. **Feature Extraction**
The system provides multiple feature extraction methods to transform input sentences into feature vectors:
- **UnigramFeatureExtractor**: Extracts bag-of-words (unigram) features.
- **BigramFeatureExtractor**: Extracts bigram (pairs of adjacent words) features.
- **BetterFeatureExtractor**: Allows advanced feature engineering, such as incorporating IDF weighting.

### 2. **Machine Learning Models**
The system includes several sentiment classification models:
- **TrivialSentimentClassifier**: A simple baseline model that always predicts positive sentiment.
- **PerceptronClassifier**: A perceptron-based binary classifier.
- **LogisticRegressionClassifier**: A logistic regression model for probability-based classification.

### 3. **Model Training**
The system supports training models using:
- **Perceptron Training**: Updates weights based on prediction errors.
- **Logistic Regression Training**: Optimizes the model using the sigmoid function and cross-entropy loss.

### 4. **Sentiment Prediction**
Trained models can classify new input sentences, predicting whether the sentiment is positive or negative.

## Code Structure

### Main Components
- **`FeatureExtractor`**: Base class for feature extractors.
  - `UnigramFeatureExtractor`: Extracts unigram features.
  - `BigramFeatureExtractor`: Extracts bigram features.
  - `BetterFeatureExtractor`: Provides flexibility for advanced feature extraction.

- **`SentimentClassifier`**: Base class for sentiment classifiers.
  - `TrivialSentimentClassifier`: Baseline classifier.
  - `PerceptronClassifier`: Implements the perceptron algorithm.
  - `LogisticRegressionClassifier`: Implements logistic regression.

- **Training Functions**:
  - `train_perceptron`: Trains a perceptron classifier.
  - `train_logistic_regression`: Trains a logistic regression classifier.

### Workflow
1. Initialize a feature extractor (e.g., unigram, bigram, or custom).
2. Choose a model (e.g., Perceptron, Logistic Regression).
3. Train the model using labeled training data.
4. Use the trained model to predict sentiment for new input sentences.

## Usage

### Prerequisites
Install the required libraries:
```bash
pip install numpy nltk tqdm
```

### Preparing Data
The input data should be a list of `SentimentExample` objects, where each example contains:
- **`words`**: A list of words in the sentence.
- **`label`**: The sentiment label (0 for negative, 1 for positive).

### Training a Model
```python
from models import train_model

# Example usage:
args = {
    "model": "LR",        # Choose model: "TRIVIAL", "PERCEPTRON", or "LR"
    "feats": "UNIGRAM"    # Choose feature type: "UNIGRAM", "BIGRAM", or "BETTER"
}
train_exs = [SentimentExample(words=["I", "love", "this"], label=1),
             SentimentExample(words=["This", "is", "terrible"], label=0)]
dev_exs = []  # Optional validation set

model = train_model(args, train_exs, dev_exs)
```

### Making Predictions
```python
sentence = ["This", "is", "amazing"]
prediction = model.predict(sentence)
print(f"Predicted sentiment: {prediction}")  # Output: 1 for positive, 0 for negative
```

## Extending the System
You can extend this system by:
1. Adding new feature extractors (subclassing `FeatureExtractor`).
2. Implementing additional machine learning models (subclassing `SentimentClassifier`).
3. Experimenting with new training techniques or optimizations.

## Example Models

### Trivial Classifier
- Always predicts positive sentiment.

### Perceptron Classifier
- Uses a weight vector to classify sentiment.
- Updates weights based on prediction errors:
  
  $w[k] = w[k] - (y_{\text{pred}} - y) \cdot \alpha \cdot \text{feature\_value}$

### Logistic Regression Classifier
- Computes probabilities using the sigmoid function:
  
  $\sigma(x) = \frac{1}{1 + e^{-x}}$

- Optimizes using cross-entropy loss.
