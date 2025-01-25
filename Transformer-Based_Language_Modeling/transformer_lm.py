# models.py

import numpy as np
import torch
import torch.nn as nn
import numpy as np
import collections 
from torch import optim, xlogy
import random
import time
from torch.utils.data import DataLoader

from transformer import PositionalEncoding
import torch.nn.functional as F

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

class TransformerEndcoder(nn.Module):
    def __init__(self, num_positions, d_model, num_hidden, num_layers, vocab_index, vocab_size=27, num_classes=27):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_hidden)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_positions)
        self.to_model = nn.Linear(num_positions, d_model)
        self.to_classes = nn.Linear(d_model, num_classes)
        self.PositionalEncoding = PositionalEncoding(d_model, num_positions=num_positions)
        self.vocab_index=vocab_index

    def forward(self, context):
        indices = []
        for c in context:
            indices.append(self.vocab_index.index_of(c))
        indices = torch.tensor(indices)
        embeddings = self.embeddings(indices)
        encoded = self.PositionalEncoding(self.to_model(embeddings))
        # pass in a triangular matrix of zeros / negative infinities
        mask = torch.triu(torch.ones(len(indices), len(indices)) * float('-inf'), diagonal=1)
        transform = self.transformer_encoder(encoded, mask)

        log_probs = F.log_softmax(self.to_classes(transform), dim=-1)
        return log_probs


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model,chunk_size,vocab_index):
        super(NeuralLanguageModel, self).__init__()
        self.model=model
        self.chunk_size=chunk_size
        self.vocab_index=vocab_index

    def get_next_char_log_probs(self, context):
        log_probs = self.model.forward(context)
        return log_probs[-1].detach().numpy()
        #raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        indices = []
        for c in next_chars:
            indices.append(self.vocab_index.index_of(c))

        indices = torch.tensor(indices)

        phrase = context + next_chars
        
        log_probs = torch.zeros(len(phrase), 27)
        chunks = len(phrase) // self.chunk_size
        ending = 0
        for i in range(chunks):
            log_probs[i * self.chunk_size : (i + 1) * self.chunk_size] = self.model.forward(phrase[i * self.chunk_size : (i + 1) * self.chunk_size])
            ending = (i + 1) * self.chunk_size

        if len(phrase[ending:]) > 0:
            log_probs[ending:]  = self.model.forward(phrase[ending:])
            
        loss = 0.0
        
        for i in range(len(phrase) - len(context)):
            loss += log_probs[len(context) - 1 + i][indices[i]]
                
        return loss.item()
        #raise Exception("Implement me")

def create_batches(data, batch_size):
    # Helper function to create batches of data
    num_batches = len(data) // batch_size
    data = data[:num_batches * batch_size]  # Trim to ensure even batch sizes
    data = np.array(data)
    data = data.reshape((batch_size, -1))
    return [(torch.LongTensor(batch), torch.LongTensor(batch[1:])) for batch in data.T]


from torch.utils.data import DataLoader, TensorDataset


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    
    chunk_size = 20
    chunks = len(train_text) // chunk_size
    #model = NeuralLanguageModel(num_positions=20, d_model=64, nhead=4, num_layers=2, indexer=vocab_index)
    model = TransformerEndcoder(num_positions=chunk_size, d_model=64, num_hidden=4, num_layers=2, vocab_index=vocab_index)
    
    model.train()
    model.zero_grad()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    chunks = len(train_text) // chunk_size

    num_epochs = 7
    for epoch in range(0, num_epochs):
        total_loss = 0.0
        num_epoches = 0
        criterion = nn.NLLLoss()
        """
        for chunk in range(chunks - 1):
            num_epoches+=1
            text = train_text[chunk * chunk_size : (chunk + 1) * chunk_size]
            context = ' ' + text[0 : chunk_size - 1]
            target = torch.LongTensor(chunk_size)

            for i in range(chunk_size):
                target[i] = vocab_index.index_of(text[i])
            log_probs = model.forward(context)
            # Calculate loss
            loss = criterion(log_probs, target)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        """
        chunks = len(train_text) // chunk_size
        for batch_num in range(chunks):
            num_epoches+=1
            start_idx = batch_num * chunk_size
            end_idx = (batch_num + 1) * chunk_size

            input_sequence = train_text[start_idx:end_idx]
            context = ' ' + input_sequence[0 : chunk_size - 1]

            target = torch.LongTensor(chunk_size)
            for i in range(chunk_size):
                target[i] = vocab_index.index_of(input_sequence[i])

            # Forward pass
            optimizer.zero_grad()
            log_probs = model(context)
            # Calculate loss
            loss = criterion(log_probs, target)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        

            
          
        total_loss/=num_epoches
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    model.eval()
    clssifier = NeuralLanguageModel(model,chunk_size=chunk_size,vocab_index=vocab_index)
    
    return clssifier

def train_lm2(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # Convert characters to indices using vocab_index
    train_indices = [vocab_index.index_of(char) for char in train_text]
    dev_indices = [vocab_index.index_of(char) for char in dev_text]


    #model = NeuralLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    #give a chunk of 20 characters
    chunk_size = 20
    chunks = len(train_text) // chunk_size
    vocab_size = len(vocab_index)

    train_data = torch.tensor(train_indices, dtype=torch.long)
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=vocab_size, shuffle=True)
    
    model = TransformerEndcoder(num_positions=chunk_size, d_model=64, num_hidden=4, num_layers=2,vocab_index=vocab_index)
    model.train()
    model.zero_grad()

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #clssifier = NeuralLanguageModel(model,chunk_size=chunk_size,vocab_index=vocab_index)
    
    time_s = time.time()
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0.0
        num_epoches = 0.0
        
        for chunk in range(chunks - 1):
            num_epoches+=1
            text = train_text[chunk * chunk_size : (chunk + 1) * chunk_size]
            #PAD 0 
            context = ' ' + text[0 : chunk_size - 1]
            gold = torch.LongTensor(chunk_size)

            for i in range(chunk_size):
                gold[i] = vocab_index.index_of(text[i])
                
            log_probs = model.forward(context)
            loss = criterion(log_probs, gold)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        """
        chunks = len(train_text) // chunk_size
        for batch_num in range(chunks):
            num_epoches+=1
            start_idx = batch_num * chunk_size
            end_idx = (batch_num + 1) * chunk_size

            input_sequence = train_text[start_idx:end_idx]
            context = ' ' + input_sequence[0 : chunk_size - 1]
            #target_sequence = train_text[start_idx + 1:end_idx + 1]
            #target_indices = [vocab_index.index_of(c) for c in target_sequence]
            #target_indices = torch.tensor(target_indices)
            target_indices = torch.LongTensor(chunk_size)
            for i in range(chunk_size):
                target_indices[i] = vocab_index.index_of(input_sequence[i])

            # Forward pass
            optimizer.zero_grad()
            output = model(input_sequence)

            # Calculate loss
            loss = criterion(output, target_indices)

            total_loss += loss
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        """

        #total_loss/=num_epoches
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    print("training time:", time.time()-time_s)

    clssifier = NeuralLanguageModel(model,chunk_size=chunk_size,vocab_index=vocab_index)
    return clssifier