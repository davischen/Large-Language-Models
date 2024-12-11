# factcheck.py

from typing import List
import numpy as np
import spacy
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))

def stem_word2(word: str) -> str:
    """
    Simple stemming function (this is not comprehensive like NLTK's PorterStemmer)
    """
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def clean_wiki_content(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove links and references (e.g., [1], [2], etc.)
    text = re.sub(r'\[[0-9]+\]', ' ', text)
    
    # Remove citations e.g. {{Cite...}}
    text = re.sub(r'{{Cite.*?}}', ' ', text)
    
    # Remove files and images e.g. [[File:...]]
    text = re.sub(r'\[\[File:.*?\]\]', ' ', text)

    # Remove "</sn"
    text = text.replace('</sn', '')

    # Remove "</sn"
    text = text.replace('/s>', '')
    
    # Remove non-alphanumeric characters except for basic punctuation and whitespace
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^A-Za-z0-9.,!?\'"`]', ' ', text)

    #text = re.sub(r'\s+', ' ', text).strip() #don't affect
    return text


def clean_wiki_content_bow(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove links and references (e.g., [1], [2], etc.)
    text = re.sub(r'\[[0-9]+\]', ' ', text)
    
    # Remove citations e.g. {{Cite...}}
    text = re.sub(r'{{Cite.*?}}', ' ', text)
    
    # Remove files and images e.g. [[File:...]]
    text = re.sub(r'\[\[File:.*?\]\]', ' ', text)

    # Remove "</sn"
    text = text.replace('</sn', '')

    # Remove "</sn"
    text = text.replace('/s>', '')
    
    # Remove non-alphanumeric characters except for basic punctuation and whitespace
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^A-Za-z0-9.,!?\'"`]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip() #don't affect

    return text

class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis:str ):
        #print('check_entailment')
        # Tokenize the premise and hypothesis
        #premise=clean_wiki_content(premise)
        #print(premise)
        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
        # Get the model's prediction
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Note that the labels are ["contradiction", "neutral", "entailment"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.
        
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the label with the highest probability
        label_idx = torch.argmax(probs, dim=-1).item()
        
        # Define the label names
        mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
        
        
        return mapping[label_idx],probs[0][label_idx].item()
        #raise Exception("Not implemented")


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction

class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):
  def __init__(self):
        self.threshold = 0.66
        self.stopwords = set(stopwords.words('english'))
    
  def tokenize(self, text: str) -> List[str]:
        # Convert to lower case and tokenize
        
        tokens = text.replace(',', '').replace('.', '')
        
        tokens = clean_wiki_content_bow(tokens)

        tokens = tokens.lower().split()

        # Optionally remove stopwords
        tokens = [token for token in tokens if token not in self.stopwords]
        
        # Stemming each token
        tokens = [PorterStemmer().stem(token) for token in tokens]

        return tokens

  def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = set(self.tokenize(fact))
        print('================')
        print(fact_tokens)
        # Combine all passages into one set of tokens
        all_passages_tokens = set()
        for passage in passages:
            text = passage['text']
            text = text.replace('</s><s>', '').replace('<s>', '').replace('</s>', '')
            each_token=self.tokenize(text)
            print(each_token)
            all_passages_tokens.update(each_token)

        # Calculate overlap
        overlap = len(fact_tokens.intersection(all_passages_tokens)) / len(fact_tokens)
        print(overlap)
        # Compare the overlap with the threshold to make a prediction
        if overlap >= self.threshold:
            return "S"
        else:
            return "NS"

def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(' '.join(words[i:i+chunk_size]))
        return chunks

def _chunk_sentence(self, sentence: str) -> List[str]:
        words = sentence.split()
        chunks = [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
        return chunks

class EntailmentFactChecker(object):
    def __init__(self, entailment_model: EntailmentModel):
      self.entailment_model = entailment_model
      self.stopwords = set(stopwords.words('english'))

    def stem_sentence(self,sentence: str) -> str:
      """
      Apply simple stemming to each word in a sentence.
      """
      # Define the suffixes to be removed
      suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 'ment']

      # Tokenize the sentence into words
      words = sentence.split()

      # Stem each word
      stemmed_words = []
      for word in words:
          for suffix in suffixes:
              if word.endswith(suffix):
                  word = word[:-len(suffix)]
                  break
          stemmed_words.append(word)

      # Join the stemmed words back into a sentence
      stemmed_sentence = ' '.join(stemmed_words)
      return stemmed_sentence

    def predict(self, fact: str, passages: List[dict]) -> str:
      max_entailment_score = float('0')
      thresold =0.54
      last_sentence=""
      #stopwords_list = ['References','Hello','Biography','Notes','Filmography','Personal life','Family','Honours','Discography']
      
      for passage in passages:
        #print(passage['text'])
        clean_text = (passage['text'])
        # reduce (X)
        #clean_text = clean_text.replace('</s><s>', '').replace('<s>', '').replace('</s>', '')
        #sentences = clean_text.split('|||')
        sentences = sent_tokenize(clean_text)
        
        for sentence in sentences:
          #for word in stopwords_list:
          #  word_pattern = r'\b' + word + r'\b'
          #  sentence = re.sub(word_pattern, '', sentence, flags=re.IGNORECASE)
          
          relation, score = self.entailment_model.check_entailment((sentence), fact)
          number_of_words = len(sentence.split())
            
          if number_of_words > 5:
          # Get entailment score (using the softmax value for the "entailment" class)
            if relation == "entailment":
              print(relation+","+str(score))
            else:
              print("[NS]"+sentence)
            if relation == "entailment" and number_of_words > 5 and score > max_entailment_score:# and score > thresold:
              max_entailment_score = score
              #print("S:"+relation+":"+str(score))
              print("[S]"+sentence)
              return "S"
      return "NS"

# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations