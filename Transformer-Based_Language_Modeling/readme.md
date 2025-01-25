
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
