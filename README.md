# NLP
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


---


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

---

# Transformer-Based Language Modeling and Sequence Labeling

## Overview
This project implements a **Transformer-based Language Model** for character-level sequence prediction and **Sequence Labeling Tasks** such as letter counting. The system uses self-attention mechanisms from the Transformer architecture to predict the next character probabilities or label sequences efficiently. Additionally, it supports visualizing attention maps to analyze how the model focuses on different parts of the input.

## Features

### 1. **Language Modeling**
- Predict the probability distribution of the next character given a context.
- Compute the log probability of a sequence of characters.
- Leverages a custom **Transformer Encoder** for sequence processing.

### 2. **Sequence Labeling**
- Assign labels (e.g., character occurrence counts) to each character in a sequence.
- Includes:
  - Multi-layer Transformer with self-attention and feed-forward networks.
  - Output predictions for each character in the input sequence.

### 3. **Transformer Architecture**
- Implements:
  - **Positional Encoding** for preserving sequence order.
  - **Transformer Layers** with self-attention and feed-forward networks.
  - Multi-layer architecture for deep sequence representation.

### 4. **Training**
- Supports training routines for:
  - Language Modeling: Optimized using **Negative Log Likelihood Loss (NLLLoss)**.
  - Sequence Labeling: Trained with labeled examples.
- Uses **Adam Optimizer** for efficient gradient-based optimization.

### 5. **Evaluation and Visualization**
- Evaluates sequence labeling accuracy.
- Provides attention map visualization for analyzing self-attention focus.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install the required libraries:
```bash
pip install torch numpy matplotlib
```

---

## How to Use

### **1. Language Model Training**
1. Prepare training and development text data as sequences of characters.
2. Create a vocabulary index for mapping characters to indices.
3. Train the model:
   ```python
   lm_model = train_lm(args, train_text, dev_text, vocab_index)
   ```
4. Predict next character probabilities:
   ```python
   log_probs = lm_model.get_next_char_log_probs("hello")
   print("Log probabilities for next character:", log_probs)
   ```
5. Compute log probabilities for a sequence:
   ```python
   sequence_log_prob = lm_model.get_log_prob_sequence("world", "hello")
   print("Log probability of sequence:", sequence_log_prob)
   ```

### **2. Sequence Labeling Training**
1. Prepare labeled examples using the `LetterCountingExample` class:
   ```python
   train_examples = [
       LetterCountingExample(input="hello", output=np.array([0, 0, 0, 1, 0]), vocab_index=vocab_index),
       ...
   ]
   ```
2. Train the sequence labeling model:
   ```python
   classifier_model = train_classifier(args, train_examples, dev_examples)
   ```
3. Evaluate the model:
   ```python
   decode(classifier_model, dev_examples, do_print=True, do_plot_attn=False)
   ```
4. Visualize attention maps (optional):
   - Save attention plots for each input example during evaluation.

---

## Model Architecture

### **Transformer Encoder**
- **Embedding Layer**: Converts characters into dense vector representations.
- **Positional Encoding**: Adds sequential information to embeddings.
- **Self-Attention Layers**:
  - Captures dependencies between input tokens.
  - Outputs attention maps for analysis.
- **Feed-Forward Networks**:
  - Non-linear transformations for richer representations.
- **Output Layer**:
  - Predicts probabilities for next characters or sequence labels.

### **Neural Language Model**
- Wraps the Transformer Encoder for:
  - Next-character prediction.
  - Sequence log-probability computation.

---

## Key Functions

### **Training**
- `train_lm`: Trains the Neural Language Model.
- `train_classifier`: Trains the Transformer for sequence labeling tasks.

### **Evaluation**
- `decode`: Evaluates sequence labeling accuracy and optionally visualizes attention maps.

### **Helper Functions**
- `create_batches`: Creates mini-batches of data for training.

---

## Applications

### **1. Character-Level Language Modeling**
- Predict the next character in a sequence.
- Score sequences of characters for language generation tasks.

### **2. Sequence Labeling**
- Label characters based on predefined rules (e.g., counting occurrences).
- Applicable to tasks like sequence tagging or classification.

### **3. Self-Attention Analysis**
- Visualize attention maps to understand how the model focuses on different parts of the input.

---
