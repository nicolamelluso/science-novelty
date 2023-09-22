from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import numpy as np
import pandas as pd

print('Reading stopwords...')
stopwords = []
with open('../data/stopwords.csv','r', encoding = 'utf-8') as file:
    for line in file:
        stopwords.append(line.replace('\n',''))
        
stopwords = set(stopwords)

import string
punctuation = string.punctuation.replace('-','')

# Method to tokenize and remove stopwords from the text
def tokenize(text):
    text = re.findall(r'[a-z0-9][a-z0-9-]*[a-z0-9]+|[a-z0-9]', text.lower())
    return text

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    
    lemmatized = [lemmatizer.lemmatize(word) for word in text]
    
    return lemmatized

def plain_text_from_inverted(inverted_index):
    
    if inverted_index is None:
        return None

    positions = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))

    positions.sort()

    return ' '.join([word for index, word in positions])


def get_removal_words(unigrams):

    return set([w for w in unigrams if ((not w.isascii()) |
                                            (len(w) == 1) |
                                            (w in punctuation) |
                                            (w.isdigit())) |
                                            (w.startswith(punctuation) |
                                            (w.startswith('-')) |
                                            (w.endswith(punctuation)) |
                                            (w.endswith('-')))] + list(stopwords))

def get_unigrams(text, processed = True):
    
    unigrams = lemmatize(tokenize(text))
    
    if processed != True:
        return unigrams
    else:
        return [w for w in unigrams if w not in get_removal_words(unigrams)]

def get_bigrams(unigrams, removal_words = None):
    
    if removal_words is None:
        removal_words = get_removal_words(unigrams)
    
    bigrams = list(ngrams(unigrams, 2))
        
    # Remove bigrams that contain at least one stopword
    return ['_'.join(bigram) for bigram in bigrams if not ((bigram[0] in removal_words) | (bigram[1] in removal_words))]

def get_trigrams(unigrams, removal_words = None):
    
    if removal_words is None:
        removal_words = get_removal_words(unigrams)
    
    trigrams = list(ngrams(unigrams, 3))
        
    # Remove trigrams that start or end with one stopword
    return ['_'.join(trigram) for trigram in trigrams if not ((trigram[0] in removal_words) | (trigram[-1] in removal_words))]

