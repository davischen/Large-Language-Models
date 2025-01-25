# Sentiment Analysis System with Deep Learning

## Overview
This project implements a **Sentiment Analysis System** using a Feedforward Neural Network (FFNN) in PyTorch. It is designed to classify sentences as having either positive or negative sentiment. The system incorporates pre-trained word embeddings and supports handling noisy data (e.g., sentences with typos).

## Features

### 1. **Base Sentiment Classifier**
- `SentimentClassifier`: A base class for defining sentiment classification models.
- Includes:
  - `predict`: Predict the sentiment of a single sentence.
  - `predict_all`: Predict the sentiment for a batch of sentences.

### 2. **Predefined Models**
- **TrivialSentimentClassifier**:
  - A simple baseline model that always predicts positive sentiment (class `1`).
- **NeuralSentimentClassifier**:
  - A wrapper for the FFNN model that handles preprocessing, embedding lookup, and prediction.

### 3. **Feedforward Neural Network (FFNN)**
- Implements the core sentiment classification logic with the following structure:
  - **Word Embedding Layer**: Optionally initializes from pre-trained word embeddings.
  - **Fully Connected Layers**: Two linear layers with ReLU activation.
  - **Output Layer**: Outputs probabilities for positive and negative classes.
- Uses Xavier initialization for weight initialization.

### 4. **Training Workflow**
- Supports mini-batch gradient descent with:
  - Cross-entropy loss.
  - Adam optimizer.
- Logs loss per epoch during training.

### 5. **Typo Handling**
- Includes the ability to toggle typo-specific preprocessing via the `has_typos` parameter during prediction.

---

## How to Use

### Training the Model
1. Prepare your data:
   - Use `SentimentExample` objects to represent sentences and their sentiment labels.
   - Example:
     ```python
     train_exs = [
         SentimentExample(words=["I", "love", "this"], label=1),
         SentimentExample(words=["This", "is", "terrible"], label=0)
     ]
     dev_exs = []  # Optional validation data
     ```

2. Load pre-trained word embeddings:
   ```python
   word_embeddings = WordEmbeddings.load_from_file("path/to/embeddings.txt")
   ```

3. Train the model:
   ```python
   args = {
       "hidden_size": 128,
       "num_epochs": 10,
       "lr": 0.001
   }
   model = train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings, train_model_for_typo_setting=True)
   ```

### Predicting Sentiment
- Predict the sentiment of a sentence:
  ```python
  prediction = model.predict(["This", "movie", "is", "amazing"], has_typos=False)
  print(f"Predicted sentiment: {prediction}")  # Output: 1 (positive sentiment)
  ```

- Batch prediction:
  ```python
  predictions = model.predict_all([
      ["I", "love", "this"],
      ["This", "is", "terrible"]
  ], has_typos=False)
  print(predictions)  # Output: [1, 0]
  ```

---

## Model Architecture
1. **Input Layer**:
   - Sentence tokens are mapped to indices based on the word embeddings.
2. **Word Embedding Layer**:
   - Maps indices to dense vector representations.
   - Averages word embeddings if a sentence has multiple words.
3. **Hidden Layers**:
   - Fully connected layers transform the input embedding into a high-dimensional feature space.
4. **Output Layer**:
   - Outputs probabilities for positive and negative classes using softmax.

---

## Key Functions

### `train_deep_averaging_network`
- Trains the FFNN model using:
  - CrossEntropyLoss for binary classification.
  - Adam optimizer for efficient gradient descent.

### `get_indices`
- Converts a batch of sentences into padded word index tensors and corresponding sentiment labels.

### `predict` (NeuralSentimentClassifier)
- Maps a sentence to its sentiment prediction by passing it through the FFNN.

---

## Applications
- **Sentiment Analysis**:
  - Analyze customer reviews, social media posts, or other textual data.
- **Typo Handling**:
  - Evaluate model robustness with noisy or typo-laden input data.
- **Custom Embeddings**:
  - Use domain-specific pre-trained word embeddings to improve performance.
