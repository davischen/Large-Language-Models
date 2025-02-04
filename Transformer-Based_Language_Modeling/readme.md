
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

## Program Overview: Transformer Model Implementation (`transformer.py`)

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

   $Q = XW^Q$  
   $K = XW^K$  
   $V = XW^V$

   where $W^Q$, $W^K$, and $W^V$ are weight matrices learned by the model. These internal computations are abstracted by the framework, simplifying the implementation.

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
## Program Overview: Language Model Implementation (`transformer_m.py`)

This program utilizes PyTorch to implement different types of **language models** with a focus on character-level modeling. It includes a uniform probability model and a Transformer-based neural language model to predict the next character in a sequence based on the context.

---

### Modules and Function Descriptions

1. **Imported Modules**:
   - `torch`, `torch.nn`, `torch.optim`: Used for building, training, and optimizing neural network models.
   - `numpy`, `random`, `collections`, `time`: Used for data manipulation, randomization, and performance measurement.
   - `torch.utils.data`: Provides utilities like `DataLoader` for handling datasets.
   - `transformer.PositionalEncoding`: Custom positional encoding class imported from the Transformer implementation.

---

### Core Model Structure and Transformer Model Relationship

1. **`LanguageModel` (Base Class)**:
   - Defines the interface for all language models, including methods to get log probabilities of the next character (`get_next_char_log_probs`) and the log probability of a sequence (`get_log_prob_sequence`).

2. **`UniformLanguageModel` (Subclass of `LanguageModel`)**:
   - Implements a simple uniform probability model where each character has an equal chance of appearing.
   - **`get_next_char_log_probs(context)`**: Returns uniform log probabilities over the vocabulary.
   - **`get_log_prob_sequence(next_chars, context)`**: Returns the log probability of a sequence assuming uniform distribution.

3. **`TransformerEndcoder` (Neural Network Model)**:
   - Implements a Transformer encoder architecture for language modeling.

   **Program and Transformer Model Correspondence:**

   - **Embedding Layer:** Converts characters into vector representations, aligning with the input embeddings of the Transformer model.
     ```python
     self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_positions)
     ```

   - **Positional Encoding:** Adds sequence position information to embeddings, allowing the model to recognize the order of characters.
     ```python
     self.PositionalEncoding = PositionalEncoding(d_model, num_positions=num_positions)
     ```

   - **Self-Attention Mechanism:** Implemented using PyTorch's `nn.TransformerEncoderLayer`, which includes multi-head self-attention and feedforward networks.
     ```python
     encoder_layer = nn.TransformerEncoderLayer(d_model, num_hidden)
     self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
     ```
     - **`d_model`:** Dimension of the model input and output.
     - **`num_hidden`:** Number of hidden units in the feedforward layer.
     - **`num_layers`:** Number of Transformer encoder layers stacked to improve model performance.

   - **Masked Attention:** A triangular mask is used to ensure that predictions for a character only consider previous characters in the sequence, mimicking the autoregressive property of language models.
     ```python
     mask = torch.triu(torch.ones(len(indices), len(indices)) * float('-inf'), diagonal=1)
     ```

   - **Output Layer:** Converts the final encoded vectors to class probabilities corresponding to each character in the vocabulary.
     ```python
     self.to_classes = nn.Linear(d_model, num_classes)
     log_probs = F.log_softmax(self.to_classes(transform), dim=-1)
     ```

   **Feedforward Neural Network (FFN) in Transformer:**
   - Each Transformer encoder layer includes a **Feedforward Neural Network (FFN)** that processes each position independently, adding non-linearity and improving the model's expressive power.
   ```python
   self.feedforward = nn.Sequential(
       nn.Linear(d_model, d_internal),
       nn.ReLU(),
       nn.Linear(d_internal, d_model)
   )
   ```
   - **Relationship to Transformer Model:**
     - The FFN operates after the self-attention mechanism in each Transformer layer.
     - It applies two linear transformations with a ReLU activation in between, allowing the model to learn complex patterns beyond simple linear relationships.
     - **Residual connections** and **layer normalization** are applied around both the self-attention and feedforward components to stabilize training and improve convergence.
     ```python
     x = input_vecs + attn_output  # Residual connection after self-attention
     x = F.layer_norm(x, x.size()[1:])  # Layer normalization
     ff_output = self.feedforward(x)  # Feedforward network
     x = x + ff_output  # Residual connection after FFN
     x = F.layer_norm(x, x.size()[1:])  # Final normalization
     ```

4. **`NeuralLanguageModel` (Subclass of `LanguageModel`)**:
   - Wraps a neural model (like `TransformerEndcoder`) to implement the language model interface.
   - **`get_next_char_log_probs(context)`**: Uses the neural model to get log probabilities for the next character.
   - **`get_log_prob_sequence(next_chars, context)`**: Calculates the total log probability of a sequence based on the given context.

   **Program and Transformer Model Correspondence:**
   - The model processes sequences autoregressively, predicting each next character based on the previous context, which is characteristic of Transformer-based language models.
   ```python
   def get_next_char_log_probs(self, context):
       log_probs = self.model.forward(context)
       return log_probs[-1].detach().numpy()
   ```

---

### Training Workflow

1. **`train_lm` Function**:
   - Trains a `TransformerEndcoder` model on character sequences.

   **Program and Transformer Model Correspondence:**
   - The training loop feeds sequences into the Transformer, optimizing it to predict the next character in a sequence, a standard language modeling task.
   ```python
   optimizer = optim.Adam(model.parameters(), lr=1e-3)
   criterion = nn.NLLLoss()

   for epoch in range(num_epochs):
       total_loss = 0.0
       for batch_num in range(chunks):
           input_sequence = train_text[batch_num * chunk_size : (batch_num + 1) * chunk_size]
           context = ' ' + input_sequence[:-1]
           target = torch.LongTensor([vocab_index.index_of(c) for c in input_sequence])

           optimizer.zero_grad()
           log_probs = model(context)
           loss = criterion(log_probs, target)
           loss.backward()
           optimizer.step()
           total_loss += loss.item()
   ```

2. **`train_lm2` Function**:
   - An alternative training function that introduces data batching using `DataLoader`, which helps improve training efficiency.

3. **`create_batches` Function**:
   - Helper function to split data into evenly sized batches and convert text data into tensors suitable for model training.

---

### Example Explanation

Assuming the training text is a sequence of characters "hello world", the model will learn to predict the next character given a context. For example:
- **Context**: "hell"
- **Expected Output**: The model should predict "o" with high probability.

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
