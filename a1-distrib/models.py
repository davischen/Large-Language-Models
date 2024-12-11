# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import re
import nltk
from nltk.corpus import stopwords 
import numpy as np
from tqdm import tqdm
import random
import string

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        #raise Exception("Must be implemented")
        self.indexer = indexer

    def get_indexer(self): 
        return self.indexer
    
    def add_features(self, sentence: List[str]):
        words = set()  # Using a set to efficiently check for word presence
        for word in sentence: 
            word = word.lower()
            if word not in words: 
                words.add(word)  # Add the word to the set
                self.indexer.add_and_get_index(word)

    def extract_features(self, sentence: List[str], add_to_indexer: bool= False) -> List[int]: 
        if add_to_indexer: 
            self.add_features(sentence)
        features = []
        count = Counter()
        for word in sentence: 
            word = word.lower()
            if self.indexer.contains(word): 
                index = self.indexer.index_of(word)
                count.update([index]) 
        return list(count.items())
    
    def vocab_size(self): 
        return len(self.indexer)


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        #raise Exception("Must be implemented")
        self.indexer = indexer
    
    def get_indexer(self): 
        return self.indexer

    def add_features(self, sentence: List[str]): 
        for i in range(len(sentence)-1): 
            word_pair = sentence[i].lower() + sentence[i+1].lower()
            self.indexer.add_and_get_index(word_pair)

    def extract_features(self, sentence:List[str], add_to_indexer: bool=False)-> List[int]: 
        if add_to_indexer: 
            self.add_features(sentence)
        count = Counter()
        for i in range(len(sentence)-1): 
            word_pair = sentence[i].lower() + sentence[i+1].lower()
             
            if self.indexer.contains(word_pair): 
                index = self.indexer.index_of(word_pair)
                count.update([index])

        return list(count.items())
    
    def vocab_size(self): 
        return len(self.indexer)


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        #raise Exception("Must be implemented")
        #FeatureExtractor.__init__(self)
        self.indexer = indexer
        self.numOfDoc =0
        self.word2docCount = {}

    def get_indexer(self):
        return self.indexer

    def idf_extractor(self, train_exs):
        self.numOfDoc = len(train_exs)
        for ex_words in train_exs:
            unique_words = set(ex_words)
            for word in unique_words:
                if word in self.word2docCount:
                    self.word2docCount[word] += 1
                else:
                    self.word2docCount[word] = 1
        '''
        for ex_words in train_exs:
            for word in ex_words.words:
                key = self.indexer.index_of(word)
                if key in self.word2docCount:
                    self.word2docCount[key]+=1
                else:
                    self.word2docCount[key]=1
        '''

    def add_features(self, ex_words: List[str]):
        # stop = stopwords.words('english')
        stop=[]
        for word in ex_words:
            # word.lower()
            if word not in stop:
                self.indexer.add_and_get_index(word)

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        count = Counter()
        if add_to_indexer: 
            self.add_features(ex_words)
        for word in ex_words: 
            word = word.lower()
            if self.indexer.contains(word): 
                index = self.indexer.index_of(word)
                count.update([index]) 
        
        return list(count.items())

    def vocab_size(self): 
        return len(self.indexer)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self,feat_extractor):
        #raise Exception("Must be implemented")
        SentimentClassifier.__init__(self)
        self.feat_extractor = feat_extractor
        self.weight_vector = np.zeros((feat_extractor.vocab_size(),))
        self.feature_dic = {}
    
    def get_features(self, ex_words: List[str]) -> List[int]: 
        ex_sent = ''.join(ex_words)
        if ex_sent not in self.feature_dic: 
            feature = self.feat_extractor.extract_features(ex_words)
            self.feature_dic[ex_sent] = feature
        else: 
            feature = self.feature_dic[ex_sent]

        return feature

    def predict(self, ex_words: List[str]) -> int: 
        features = self.get_features(ex_words)
        score = sum(self.weight_vector[key]* val for key,val in features)
        return 1 if score >= 0.5 else 0

    def update_weight(self, ex_words, y, y_pred, alpha): 
        if y != y_pred:
            true_features = self.get_features(ex_words)
            for key,val in true_features:
               self.weight_vector[key] = self.weight_vector[key] - (y_pred - y) * alpha * val
                
def sigmoid(x):
      result = 1./(1. + np.exp(-x))
      return result

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor):
        #raise Exception("Must be implemented")
        SentimentClassifier.__init__(self)
        self.feat_extractor = feat_extractor
        self.weight_vector = np.zeros((feat_extractor.vocab_size(),))
        self.feature_dic = {}

    def get_features(self, sentence: List[str]) -> List[int]:
        ex_sent = ''.join(sentence)
        if ex_sent not in self.feature_dic: 
            feature = self.feat_extractor.extract_features(sentence)
            self.feature_dic[ex_sent] = feature
        else: 
            feature = self.feature_dic[ex_sent]

        return feature

    def predict(self, ex_words: List[str]) -> int: 
        features = self.get_features(ex_words)
        score = sum(self.weight_vector[key]* val for key,val in features)
        result = 1 / (1 + np.exp(-score))  # Sigmoid function sigmoid(score)
        y_pred = 1 if result>0.5 else 0
        return y_pred

    def update_weight(self, sentence, y, y_pre, alpha): 
        features = self.get_features(sentence)

        score = sum(self.weight_vector[key]* val for key,val in features)
        result = 1 / (1 + np.exp(-score))  # Sigmoid function sigmoid(score)
        
        for k , val in features:
            self.weight_vector[k] = self.weight_vector[k] - alpha * ((result - y)* val)
            #self.weights[k] = self.weights[k] - alpha * (-y * val * (1-result) + (1-result) * val * result)

    def loss(self, train_exs): 
        sum_loss = 0
        feature_list = (self.get_features(ex.words) for ex in train_exs)
        score = sum(self.weight_vector[key]* val for key,val in feature_list)
        
        result = 1 / (1 + np.exp(-score))  # Sigmoid function sigmoid(score)
        loss = -y * np.log(result) - (1-y)* np.log(1-result)
        sum_loss += loss
        sum_loss = sum_loss / float(len(train_exs))

        return sum_loss


class LogisticRegressionClassifier2(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self,feat_extractor):
        #raise Exception("Must be implemented")
        SentimentClassifier.__init__(self)
        self.feat_extractor = feat_extractor
        self.weight_vector = np.zeros((feat_extractor.vocab_size(),))
        self.feature_dic = {}

    def get_features(self, sentence: List[str]) -> List[int]:
        ex_sent = ''.join(sentence)
        if ex_sent not in self.feature_dic: 
            feature = self.feat_extractor.extract_features(sentence)
            self.feature_dic[ex_sent] = feature
        else: 
            feature = self.feature_dic[ex_sent]

        return feature

    def predict(self, ex_words: List[str]) -> int: 
        features = self.get_features(ex_words)
        score = sum(self.weight_vector[key]* val for key,val in features)
        result = 1 / (1 + np.exp(-score))  # Sigmoid function sigmoid(score)
        y_pred = 1 if result>0.5 else 0
        return y_pred

    def update_weight(self, ex_words, y, y_pred, alpha): 

        if y != y_pred:
            true_features = self.get_features(ex_words)
            for key,val in true_features:
               self.weight_vector[key] = self.weight_vector[key] - (y_pred - y) * alpha * val
        
    def loss(self, train_exs): 
        sum_loss = 0
        features = (self.get_features(ex.words) for ex in train_exs)
        score = sum(self.weight_vector[key]* val for key,val in features)
        
        result = 1 / (1 + np.exp(-score))  # Sigmoid function sigmoid(score)
        loss = -y * np.log(result) - (1-y)* np.log(1-result)
        sum_loss += loss
        sum_loss = sum_loss / float(len(train_exs))

        return sum_loss




def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    #raise Exception("Must be implemented")
    for ex in train_exs: 
        feat_extractor.add_features(ex.words)

    model = PerceptronClassifier(feat_extractor)
    epochs = 20
    alpha = 1
    for i in tqdm(range(epochs)): 
        random.shuffle(train_exs)
        data_size = int(len(train_exs))
        data_ex = train_exs[:data_size]

        for ex in data_ex: 
            y = ex.label
            y_pre = model.predict(ex.words)
            model.update_weight(ex.words, y, y_pre, alpha)
        
        alpha = alpha * 0.9
    
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    #raise Exception("Must be implemented")
    for ex in train_exs: 
        feat_extractor.add_features(ex.words)

    model = LogisticRegressionClassifier(feat_extractor)
    epochs = 30
    alpha = 0.5
   
    for i in tqdm(range(epochs)): 
        """ if(isinstance(feat_extractor, BetterFeatureExtractor)): 
            alpha = alpha / (i+1) """
        random.shuffle(train_exs)
        data_size = int(len(train_exs))
        data_exs = train_exs[:data_size]

        for ex in data_exs: 
            y = ex.label
            y_pre = model.predict(ex.words)
            model.update_weight(ex.words, y, y_pre, alpha)
        
    return model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model