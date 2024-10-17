frimport os
import csv
import sys
import re
import spacy
import itertools
import string
import pandas as pd
import requests
import nltk
from typing import List, Union, Iterator
from spacy.tokens import Doc, Span, Token
from spacy.symbols import NOUN, PROPN, PRON
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.tokenizer import Tokenizer
from spacy.language import Language
from nltk.corpus import stopwords as nltk_stopwords
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
from spacy.lang import char_classes
from spacy.matcher import Matcher
from spacy.util import filter_spans
from tqdm.notebook import tqdm
nltk.download('stopwords')

# Load the large English model from spaCy
#nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('en_core_web_sm')

nlp.max_length = 10000000  # Increase maximum length for large documents


# 1. Download and process additional stopwords from Zenodo
#zenodo_url = 'https://zenodo.org/records/13869486/files/stopwords.csv?download=1'
#response = requests.get(zenodo_url)
#with open('stopwords.csv', 'wb') as f:
#    f.write(response.content)

# Load the stopwords CSV
cleaning = pd.read_csv('https://zenodo.org/records/13869486/files/stopwords.csv?download=1')

# Extract stopwords and removals from the file
stopwords = set(cleaning[cleaning['Type'] == 1]['Word'].unique())
removals = set(cleaning[cleaning['Type'] == 2]['Word'].unique())

# Customize stopwords further
stopwords.remove('anti')
stopwords = {s for s in stopwords if len(s) > 1}

# 2. Combine stopwords with spaCy, Gensim, and NLTK
stopwords_spacy = nlp.Defaults.stop_words
stopwords_gensim = list(gensim_stopwords)
stopwords_nltk = nltk_stopwords.words('english')

# Combine all stopwords
combined_stopwords = set(itertools.chain(stopwords_spacy, stopwords_gensim, stopwords_nltk))
combined_stopwords.update(stopwords)

# Add all stopwords to spaCy's stopword list
for word in combined_stopwords:
    nlp.Defaults.stop_words.add(word)

# Extended punctuation for removal
# Be careful to remove hyphens '-'
extended_punctuation = r""".—!–"#$%&'()*+,./:;<=>?@[\]^_`{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿‽⁇⁈⁉‽⁇⁈⁉。。、、，、．．・・：：；；！！？？｡｡｢｢｣｣､､･･"""

# 4. Custom tokenizer with rules to handle custom infixes, prefixes, and suffixes
def custom_tokenizer(nlp: Language) -> Tokenizer:
    infixes = (
        char_classes.LIST_ELLIPSES
        + char_classes.LIST_ICONS
        + [r"×", r"(?<=[0-9])[+\-\*^](?=[0-9-])", r"(?<=[a-z])\.(?=[A-Z])",
           r"(?<=[a-z]),(?=[a-z])", r"(?<=[a-z])[:<>=/](?=[a-z])"]
    )

    prefixes = (
        ["§", "%", "=", r"\+"]
        + char_classes.split_chars(char_classes.PUNCT)
        + char_classes.LIST_QUOTES
        + char_classes.LIST_CURRENCY
        + char_classes.LIST_ICONS
    )

    suffixes = (
        char_classes.split_chars(char_classes.PUNCT)
        + char_classes.LIST_ELLIPSES
        + char_classes.LIST_QUOTES
        + char_classes.LIST_ICONS
    )

    infix_re = compile_infix_regex(infixes)
    prefix_re = compile_prefix_regex(prefixes)
    suffix_re = compile_suffix_regex(suffixes)

    return Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match
    )

# Replace the tokenizer with the custom one
nlp.tokenizer = custom_tokenizer(nlp)

# Initialize the matcher
matcher = Matcher(nlp.vocab)

# Add patterns for the matcher
patterns = [
    [{"POS": "VERB", "TAG": "VBG"}, {"POS": "ADJ", "IS_STOP": False, "OP": "*"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
    [{"POS": "ADJ", "IS_STOP": False, "OP": "*"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
    [{"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}]
]
matcher.add("COMBINED_PATTERN", patterns)


def has_consecutive_hyphens(text):
    # Loop through the string and check for consecutive hyphens
    for i in range(len(text) - 1):
        if text[i] == '-' and text[i + 1] == '-':
            return True
    return False

# Extract and format matches
def extract_and_format_matches(doc: Doc) -> str:
    """
    Extract and format matches using the Matcher, filtering out punctuation and applying rules for leading/trailing stopwords and removals.

    @param doc: A spaCy Doc object
    @return: A string of cleaned, filtered noun phrases
    """
    matches = matcher(doc)
    spans = filter_spans([doc[start:end] for _, start, end in matches])

    filtered_phrases = []
    
    for span in spans:
        # Lemmatize each token in the span
        lemmatized_phrase = []
        span = span.text.split()

        #print(span)
        for token in span:
            #print(f'\t{token}')
            # Handle mistakenly spelled words
            if token.startswith('-'):
                continue
            if token.endswith('-'):
                continue

            # Remove punctuation
            token = ''.join([s for s in token if s not in extended_punctuation])

            token = token.replace('\\','').lower()

            if has_consecutive_hyphens(token):
                continue

            if len(token) <= 1:
                continue
                
            # Handle hyphenated words by splitting them and lemmatizing each subword
            if '-' in token:
                subwords = token.split('-')
                subwords_lemmatized = [nlp(sub)[0].lemma_ for sub in subwords]
                
                # If any part of the hyphenated word is a stopword, skip the whole word
                if any(sub in stopwords for sub in subwords_lemmatized):
                    continue

                # If any part of the hyphenated word is a stopword, skip the whole word
                if all(sub in removals for sub in subwords_lemmatized):
                    continue

                # Re-create the lemmatized word
                subword_lemmatized = '-'.join(subwords_lemmatized)

                # Add the clean word to the phrase
                lemmatized_phrase.append(subword_lemmatized)
            else:
                # Lemmatize the word and add to the phtase
                lemmatized_phrase.append(nlp(token)[0].lemma_)

        # Remove phrase if it contains stopwords or consists entirely of removal words
        if not lemmatized_phrase or all(w in removals for w in lemmatized_phrase):
            continue

        # Remove leading stopwords
        lemmatized_phrase = [w for i, w in enumerate(lemmatized_phrase) if not (w in stopwords and all(x in stopwords for x in lemmatized_phrase[:i]))]

        # Remove trailing stopwords
        lemmatized_phrase = [w for i, w in enumerate(lemmatized_phrase) if not (w in stopwords and all(x in stopwords for x in lemmatized_phrase[i:]))]

        if lemmatized_phrase:
            if lemmatized_phrase != []:
                filtered_phrases.append(lemmatized_phrase)

    # Join the valid, cleaned noun phrases
    return ' '.join(['_'.join(phrase) for phrase in filtered_phrases if phrase])

# 7. Process text, check against stopwords and removals, and return cleaned string
def process_text(text: str) -> str:
    """
    Process text to extract noun phrases, clean and filter them according to custom rules.

    @param text: A string containing raw text
    @return: A cleaned, processed string with filtered noun chunks
    """
    doc = nlp(text)
    return extract_and_format_matches(doc)


def plain_text_from_inverted(inverted_index):
    
    if inverted_index is None:
        return None

    positions = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))

    positions.sort()

    return ' '.join([word for index, word in positions])
