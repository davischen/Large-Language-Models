# Fact-Checking System

## Overview
This project implements a **fact-checking system** to validate claims against textual evidence using various methods, ranging from simple heuristics to advanced machine learning models. It determines whether a fact is **Supported (S)**, **Not Supported (NS)**, or **Irrelevant (IR)** based on textual passages provided as evidence.

## Features

### **1. Fact Representation**
- Each fact is represented as a `FactExample` object containing:
  - **Fact**: The claim to be validated.
  - **Passages**: A list of evidence texts, each containing a title and body text.
  - **Label**: Ground truth labels (S, NS, or IR).

### **2. Fact-Checking Methods**
- **Random Guessing**:
  - Randomly predicts `S` or `NS`.
- **Always Supported**:
  - Always predicts `S` (supported).
- **Word Overlap**:
  - Computes token overlap between the fact and passages.
  - Predicts `S` if the overlap exceeds a threshold (default: 66%).
- **Entailment Model**:
  - Uses pretrained models like **DeBERTa-v3-base** or **RoBERTa-large-MNLI** to compute entailment probabilities.
  - Evaluates whether the passage entails the fact.
- **Dependency Parsing** (Optional):
  - Analyzes grammatical relations (e.g., subject-verb-object) between the fact and passages to determine alignment.

### **3. Text Preprocessing**
- Cleans and preprocesses passages to remove:
  - HTML tags.
  - References (e.g., `[1]`, `[2]`).
  - Non-alphanumeric characters.
- Provides tokenization, stemming, and stopword removal for overlap-based methods.

### **4. Evaluation**
- Generates a confusion matrix.
- Calculates:
  - **Accuracy**.
  - **Precision**, **Recall**, and **F1 Score** for `S` and `NS` classes.

---

## Installation

### Prerequisites
- Python 3.8+
- Required Libraries:
  - PyTorch
  - Transformers
  - SpaCy
  - NumPy
  - NLTK

Install the dependencies:
```bash
pip install torch transformers spacy numpy nltk
```

Download SpaCy's English model:
```bash
python -m spacy download en_core_web_sm
```

---

## How to Use

### **1. Run Fact-Checking**
Use the command-line interface to run the fact-checking system.

#### **Command Format**:
```bash
python factcheck.py --mode [random|always_entail|word_overlap|parsing|entailment]
```

#### **Arguments**:
- `--mode`: The fact-checking method to use. Options:
  - `random`: Random guessing.
  - `always_entail`: Always predicts `S`.
  - `word_overlap`: Token overlap-based fact-checking.
  - `parsing`: Dependency parsing (optional, not fully implemented).
  - `entailment`: Uses a pretrained entailment model.
- `--labels_path`: Path to the labeled facts (default: `data/dev_labeled_ChatGPT.jsonl`).
- `--passages_path`: Path to the passages used as evidence (default: `data/passages_bm25_ChatGPT_humfacts.jsonl`).

#### **Example Command**:
```bash
python factcheck.py --mode entailment --labels_path data/dev_labeled_ChatGPT.jsonl --passages_path data/passages_bm25_ChatGPT_humfacts.jsonl
```

### **2. Input Files**
- **Labels File**: JSONL file containing human-annotated facts and their ground truth labels.
- **Passages File**: JSONL file containing passages retrieved as evidence for each fact.

### **3. Output**
- Confusion matrix showing prediction results.
- Accuracy, precision, recall, and F1 scores for `S` and `NS` classes.

---

## Fact-Checking Methods in Detail

### **1. Word Overlap**
- **Description**:
  - Computes token overlap between the fact and all passages.
  - If overlap exceeds a threshold (default: 66%), predicts `S`.

- **Implementation**:
  - Tokenizes and stems both the fact and passages.
  - Removes stopwords to focus on meaningful content.

### **2. Entailment**
- **Description**:
  - Uses pretrained models (e.g., DeBERTa, RoBERTa) to evaluate entailment relationships between the fact and passages.

- **Process**:
  1. Tokenizes the fact and passage.
  2. Feeds them into a pretrained entailment model.
  3. Computes entailment probabilities.

- **Output**:
  - Predicts `S` if entailment probability exceeds a threshold (default: 0.54).

### **3. Random Guess and Always Supported**
- **Random Guess**:
  - Randomly assigns `S` or `NS` to facts.
- **Always Supported**:
  - Always predicts `S` for all facts.

---

## Example Workflow

### **Training and Fact-Checking with Entailment Model**
1. **Load Passages**:
   ```python
   fact_to_passage_dict = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
   ```
2. **Load Labeled Facts**:
   ```python
   examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", fact_to_passage_dict)
   ```
3. **Run Fact-Checking**:
   ```python
   fact_checker = EntailmentFactChecker(EntailmentModel(pretrained_model, tokenizer))
   predict_two_classes(examples, fact_checker)
   ```

---

## Applications

### **1. Fact Verification**
- Validates claims using textual evidence from Wikipedia or other sources.

### **2. Document Consistency Checking**
- Ensures that claims in documents are consistent with referenced evidence.

### **3. Pretrained Model Evaluation**
- Fine-tunes and evaluates entailment models on domain-specific fact-checking tasks.
