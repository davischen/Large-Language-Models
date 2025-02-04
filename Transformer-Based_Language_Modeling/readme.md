
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

### Program Overview: Transformer Model Implementation (`transformer.py`)

This program primarily uses the PyTorch framework to implement a simplified **Transformer model** designed for a letter counting classification task. The model determines how many times each letter in an input string has previously appeared before its current occurrence and classifies it as 0 (never appeared), 1 (appeared once), or 2 (appeared two or more times).

---

### Modules and Function Descriptions

1. **Imported Modules**:
   - `torch`, `torch.nn`, `torch.optim`: Used to build neural network models and optimize them.
   - `numpy`, `random`: Used for data manipulation and randomization.
   - `matplotlib.pyplot`: Used to visualize attention mechanisms through heatmaps.
   - `utils`: Likely contains helper functions, such as index management.

2. **`LetterCountingExample` Class**:
   - Wraps training examples, including the raw input string, indexed letters, tensorized input, and corresponding output labels (0, 1, 2).

---

### Core Model Structure

1. **`Transformer` Class**:
   - **Embedding Layer**: Converts letters into vector representations.
   - **Positional Encoding**: Adds position information to the embeddings to retain sequence order.
   - **Multiple Transformer Layers**: Each layer includes self-attention mechanisms and feedforward neural networks.
   - **Output Layer**: Converts the final output into a 3-class classification result (0, 1, 2).
   - **Forward Pass**: Passes input data through embeddings, positional encoding, and multiple Transformer layers to produce classification results and attention maps.

2. **`TransformerLayer` Class**:
   - **Self-Attention Mechanism (`MultiheadAttention`)**: Captures dependencies between different positions in the sequence.
   - **Feedforward Neural Network**: Further learns feature representations.
   - **Residual Connections and Layer Normalization**: Stabilizes training and accelerates convergence.

   **PyTorch Abstraction:**  
   In the self-attention part of the `TransformerLayer`, the following code is used:

   ```python
   self.self_attention = nn.MultiheadAttention(d_model, num_heads=1)
   attn_output, attn_map = self.self_attention(input_vecs, input_vecs, input_vecs)
   ```

   Here, `input_vecs` is passed as **Query (Q)**, **Key (K)**, and **Value (V)** to `nn.MultiheadAttention`. PyTorch automatically computes:

   \( Q = XW^Q \)  
   \( K = XW^K \)  
   \( V = XW^V \)

   where \( W^Q \), \( W^K \), and \( W^V \) are weight matrices learned by the model. These internal computations are abstracted by the framework, simplifying the implementation.

---

### Program and Transformer Model Correspondence

1. **Self-Attention Mechanism Implementation**  
   In the `TransformerLayer` class, PyTorch's `nn.MultiheadAttention` is used to implement the self-attention mechanism:

   ```python
   self.self_attention = nn.MultiheadAttention(d_model, num_heads=1)
   ```

   - **`d_model`**: Represents the feature dimension of inputs and outputs.
   - **`num_heads=1`**: This sets up single-head attention (a simplified version), while multi-head attention is a characteristic feature of the full Transformer.

2. **Positional Encoding Implementation**  
   The `PositionalEncoding` class is responsible for adding positional encoding to retain the sequence order of input data:

   ```python
   self.PositionalEncoding = PositionalEncoding(d_model=d_model)
   ```

   This positional encoding is added to the embedding vectors, allowing the model to recognize the position of each letter in the sequence.

3. **Multiple Transformer Layers**  
   The `Transformer` class uses multiple `TransformerLayer` instances to stack the model:

   ```python
   self.transformer_layers = nn.ModuleList([
       TransformerLayer(d_model=d_model, d_internal=d_internal) for _ in range(num_layers)
   ])
   ```

   - **`num_layers`**: Determines the number of Transformer layers, which is a common technique to improve model performance.

4. **Residual Connections and Normalization**  
   Each `TransformerLayer` includes residual connections and layer normalization, essential features of the Transformer model:

   ```python
   x = input_vecs + attn_output
   x = F.layer_norm(x, x.size()[1:])
   ```

   This ensures that important information is not lost during multi-layer propagation and stabilizes training.

5. **Classification Task Design**  
   Finally, the output of the Transformer model passes through a fully connected layer (`nn.Linear`) to convert high-dimensional vectors into 3-class classification results (0, 1, 2):

   ```python
   self.to_classes = nn.Linear(d_model, num_classes)
   ```

---

### Training and Testing Workflow

1. **`train_classifier` Function**:
   - Initializes the Transformer model and uses the **Adam optimizer**.
   - The loss function is **Negative Log-Likelihood Loss (`NLLLoss`)**.
   - Runs multiple training epochs, shuffles training data each round, performs forward passes, calculates loss, backpropagates, and updates model parameters.

2. **`decode` Function**:
   - Uses the trained model to make predictions on test data.
   - Calculates model accuracy and optionally prints each sample's input, actual labels, and predicted results.
   - Optionally plots attention maps to visualize which positions the model focuses on during prediction.

---

### Example Explanation

Assuming the input string is `"hello"`, the model will output for each letter:
- **h**: Never appeared before → 0
- **e**: Never appeared before → 0
- **l**: Never appeared before → 0
- **l**: Appeared once before → 1
- **o**: Never appeared before → 0

The model learns how to determine these counts and outputs the corresponding classification results.

---

### Potential Application Scenarios

- **Letter or Word Counting**: The model architecture can be extended to more complex sequence labeling tasks, such as Named Entity Recognition (NER).
- **Natural Language Processing (NLP)**: This model can serve as a foundation for learning Transformer architecture, further applied to translation, text generation, etc.
- **Sequence Analysis**: Any task that requires capturing sequence dependencies, such as DNA sequence analysis or time-series forecasting.

---

This is the main content and structure of the program. If you need further explanations on specific parts or want to see the results, feel free to let me know!
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
