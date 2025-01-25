# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,word_embeddings=None):
        super(FFNN, self).__init__()
        if word_embeddings is not None:
            self.embeddings =nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors), freeze=False)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        if self.embeddings is not None :
            word_embedding = self.embeddings(x) 
            mean = torch.mean(word_embedding, dim=1, keepdim=False).float()
            return self.fc2(self.relu(self.fc1(mean)))
        else:
            return self.fc2(self.relu(self.fc1(x)))

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model: nn.Module, word_embeddings: WordEmbeddings):
        SentimentClassifier.__init__(self)
        self.word_indexer = word_embeddings.word_indexer
        self.model = model
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str], has_typos: bool):
        # Convert words to word embeddings
        words_idx = [max(1, self.word_indexer.index_of(word)) for word in ex_words]

        # Convert the list of word embeddings to a tensor
        words_tensor = torch.tensor([words_idx])

        y_probs = self.model.forward(words_tensor)
        return torch.argmax(y_probs)

def get_indices(train_exs: List[SentimentExample], word_embeddings: WordEmbeddings):
    """
    Get the indices of a batch of Sentiment examples EXS
    """

    word_indexer = word_embeddings.word_indexer
    max_len = max(len(ex.words) for ex in train_exs)
    idxs = torch.zeros(len(train_exs), max_len, dtype=torch.long)
    labels = []

    for i, ex in enumerate(train_exs):
        ex_len = len(ex.words)
        for j in range(ex_len):
            word = ex.words[j]
            
            # Get the word index from the word indexer
            idx = word_indexer.index_of(word)
            
            # Use 1 for UNK (unknown) words if the word is not found
            idx = idx if idx != -1 else 1
            idxs[i][j] = idx

        # Use 0 for PAD (padding) tokens for the remaining positions
        for j in range(ex_len, max_len):
            idxs[i][j] = 0
        
        labels.append(ex.label)

    return torch.LongTensor(idxs), torch.LongTensor(labels)


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    batch_size = 64

     # Initialize your FFNN model
    input_size = word_embeddings.get_embedding_length()
    hidden_size = args.hidden_size
    output_size = 2  # Assuming binary classification (positive vs. negative)

    hidden_layer_size = args.hidden_size
    model = FFNN(input_size, hidden_size, output_size,word_embeddings)

    epochs = args.num_epochs
    batch_size= 128
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ex_indices = [idx for idx in range(0,len(train_exs))]
    epochs=15
    num_epochs = args.num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        np.random.shuffle(train_exs)
        total_loss=0.0
        for i in range(0, len(train_exs), batch_size):
            exs = train_exs[i : i+batch_size]
            idxs, labels = get_indices(exs, word_embeddings)
            model.zero_grad()
            probs_outputs = model.forward(idxs)
            loss = criterion(probs_outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        total_loss /= len(train_exs)
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    return NeuralSentimentClassifier(model,word_embeddings)

