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


import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def tokenize(text):
    return re.findall(r'[a-z0-9][a-z0-9-]*[a-z0-9]+|[a-z0-9]', text.lower())

def lemmatize(text):
    return [lemmatizer.lemmatize(word) for word in text]

def get_removal_words(unigrams):
    return set([w for w in unigrams if (
        not w.isascii() or
        len(w) == 1 or
        w in punctuation or
        w.isdigit() or
        w.startswith(('-', *punctuation)) or
        w.endswith(('-', *punctuation))
    )] + list(stopwords))

def process_text(text):
    unigrams = lemmatize(tokenize(text))
    removal_words = get_removal_words(unigrams)
    
    processed_unigrams = [w for w in unigrams if w not in removal_words]
    bigrams = ['_'.join(bigram) for bigram in ngrams(unigrams, 2) if not (bigram[0] in removal_words or bigram[1] in removal_words)]
    trigrams = ['_'.join(trigram) for trigram in ngrams(unigrams, 3) if not (trigram[0] in removal_words or trigram[-1] in removal_words)]
    
    return processed_unigrams, bigrams, trigrams


def plain_text_from_inverted(inverted_index):
    
    if inverted_index is None:
        return None

    positions = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))

    positions.sort()

    return ' '.join([word for index, word in positions])
