import os
import csv
import sys
import re
import spacy
import itertools
import string
import pandas as pd
import requests
import nltk
import numpy as np
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
from typing import List
from spacy.lang import char_classes
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.language import Language
from spacy.tokens import Token
from scispacy.consts import ABBREVIATIONS

nltk.download('stopwords')

# Load the English model from spaCy
nlp = spacy.load('en_core_web_sm')
## In the paper we use the "en_core_sci_lg"
## For demonstration purposes, in this repository we load the small one.

nlp.max_length = 10000000  # Increase maximum length for large documents

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
nlp.max_length = 1000000000

import string
punctuation = string.punctuation.replace('-','')

def remove_new_lines(text: str) -> str:
    """Used to preprocess away new lines in the middle of words. This function
       is intended to be called on a raw string before it is passed through a
       spaCy pipeline

    @param text: a string of text to be processed
    """
    text = text.replace("-\n\n", "")
    text = text.replace("- \n\n", "")
    text = text.replace("-\n", "")
    text = text.replace("- \n", "")
    return text


def combined_rule_prefixes() -> List[str]:
    """Helper function that returns the prefix pattern for the tokenizer.
    It is a helper function to accomodate spacy tests that only test
    prefixes.
    """
    # add lookahead assertions for brackets (may not work properly for unbalanced brackets)
    prefix_punct = char_classes.PUNCT.replace("|", " ")
    prefix_punct = prefix_punct.replace(r"\(", r"\((?![^\(\s]+\)\S+)")
    prefix_punct = prefix_punct.replace(r"\[", r"\[(?![^\[\s]+\]\S+)")
    prefix_punct = prefix_punct.replace(r"\{", r"\{(?![^\{\s]+\}\S+)")

    prefixes = (
        ["§", "%", "=", r"\+"]
        + char_classes.split_chars(prefix_punct)
        + char_classes.LIST_ELLIPSES
        + char_classes.LIST_QUOTES
        + char_classes.LIST_CURRENCY
        + char_classes.LIST_ICONS
    )
    return prefixes


def combined_rule_tokenizer(nlp: Language) -> Tokenizer:
    """Creates a custom tokenizer on top of spaCy's default tokenizer. The
    intended use of this function is to replace the tokenizer in a spaCy
    pipeline like so:

         nlp = spacy.load("some_spacy_model")
         nlp.tokenizer = combined_rule_tokenizer(nlp)

    @param nlp: a loaded spaCy model
    """
    # remove the first hyphen to prevent tokenization of the normal hyphen
    hyphens = char_classes.HYPHENS.replace("-|", "", 1)

    infixes = (
        char_classes.LIST_ELLIPSES
        + char_classes.LIST_ICONS
        + [
            r"×",  # added this special x character to tokenize it separately
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}])\.(?=[{au}])".format(
                al=char_classes.ALPHA_LOWER, au=char_classes.ALPHA_UPPER
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=char_classes.ALPHA),
            r'(?<=[{a}])[?";:=,.]*(?:{h})(?=[{a}])'.format(
                a=char_classes.ALPHA, h=hyphens
            ),
            # removed / to prevent tokenization of /
            r'(?<=[{a}"])[:<>=](?=[{a}])'.format(a=char_classes.ALPHA),
        ]
    )

    prefixes = combined_rule_prefixes()

    # add the last apostrophe
    quotes = char_classes.LIST_QUOTES.copy() + ["’"]

    # add lookbehind assertions for brackets (may not work properly for unbalanced brackets)
    suffix_punct = char_classes.PUNCT.replace("|", " ")
    # These lookbehinds are commented out because they are variable width lookbehinds, and as of spacy 2.1,
    # spacy uses the re package instead of the regex package. The re package does not support variable width
    # lookbehinds. Hacking spacy internals to allow us to use the regex package is doable, but would require
    # creating our own instance of the language class, with our own Tokenizer class, with the from_bytes method
    # using the regex package instead of the re package
    # suffix_punct = suffix_punct.replace(r"\)", r"(?<!\S+\([^\)\s]+)\)")
    # suffix_punct = suffix_punct.replace(r"\]", r"(?<!\S+\[[^\]\s]+)\]")
    # suffix_punct = suffix_punct.replace(r"\}", r"(?<!\S+\{[^\}\s]+)\}")

    suffixes = (
        char_classes.split_chars(suffix_punct)
        + char_classes.LIST_ELLIPSES
        + quotes
        + char_classes.LIST_ICONS
        + ["'s", "'S", "’s", "’S", "’s", "’S"]
        + [
            r"(?<=[0-9])\+",
            r"(?<=°[FfCcKk])\.",
            r"(?<=[0-9])(?:{})".format(char_classes.CURRENCY),
            # this is another place where we used a variable width lookbehind
            # so now things like 'H3g' will be tokenized as ['H3', 'g']
            # previously the lookbehind was (^[0-9]+)
            r"(?<=[0-9])(?:{u})".format(u=char_classes.UNITS),
            r"(?<=[0-9{}{}(?:{})])\.".format(
                char_classes.ALPHA_LOWER, r"%²\-\)\]\+", "|".join(quotes)
            ),
            # add |\d to split off the period of a sentence that ends with 1D.
            r"(?<=[{a}|\d][{a}])\.".format(a=char_classes.ALPHA_UPPER),
        ]
    )

    infix_re = compile_infix_regex(infixes)
    prefix_re = compile_prefix_regex(prefixes)
    suffix_re = compile_suffix_regex(suffixes)

    # Update exclusions to include these abbreviations so the period is not split off
    exclusions = {
        abbreviation: [{ORTH: abbreviation}] for abbreviation in ABBREVIATIONS
    }
    tokenizer_exceptions = nlp.Defaults.tokenizer_exceptions.copy()
    tokenizer_exceptions.update(exclusions)

    tokenizer = Tokenizer(
        nlp.vocab,
        tokenizer_exceptions,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match,  # type: ignore
    )
    return tokenizer


# Define the custom attribute function
def is_custom_filtered(token):
    return (
        not token.is_digit and
        token.is_ascii and
        not token.is_punct and
        not token.lemma_ in nlp.Defaults.stop_words and
        token.shape_ != 'x' and
        token.shape_ != 'X' and
        not token.like_num and
        (token.shape_ and all(c in 'xXdD-' for c in token.shape_)) and
        token.text[0] != '-' and
        token.text[-1] != '-'
    )

# Set the custom attribute
Token.set_extension('is_custom_filtered', getter=is_custom_filtered, force=True)

def custom_noun_chunks(doclike: Union[Doc, Span]) -> Iterator[Span]:
    """
    Detect base noun phrases from a dependency parse, excluding initial stop words. Works on both Doc and Span.
    """
    labels = [
        "oprd",
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
    ]
    doc = doclike.doc  # Ensure works on both Doc and Span.
    if not doc.has_annotation("DEP"):
        raise ValueError("Dependency parse has not been provided")
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    np_label = doc.vocab.strings.add("NP")
    prev_end = -1
    for i, word in enumerate(doclike):
        if word.pos not in (NOUN, PROPN, PRON):
            continue
        if word.left_edge.i <= prev_end:
            continue
        if word.dep in np_deps:
            start_index = word.left_edge.i
            # Skip initial stop words
            while doc[start_index].is_stop:
                start_index += 1
                if start_index == word.i + 1:   # All tokens are stop words -- unlikely but safe to check
                    break
            prev_end = word.i
            yield doc[start_index: word.i + 1]#, np_label
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            if head.dep in np_deps:
                start_index = word.left_edge.i
                # Skip initial stop words
                while doc[start_index].is_stop:
                    start_index += 1
                    if start_index == word.i + 1:  # All tokens are stop words -- unlikely but safe to check
                        break
                prev_end = word.i
                yield doc[start_index: word.i + 1]#, np_label

from spacy.tokens import Doc
from typing import Iterable, Iterator

nlp = spacy.load('en_core_web_lg')
nlp.max_length = 10000000

# Replace the tokenizer with the custom one
nlp.tokenizer = combined_rule_tokenizer(nlp)

import string
# Function to check if a token contains punctuation
valid_chars = string.punctuation.replace('-', '')  # Remove '-' from the list of punctuation characters  
from spacy.matcher import Matcher
from spacy.util import filter_spans
# Create the Matcher object
matcher = Matcher(nlp.vocab)

# Define patterns with proper nouns included
patterns = [
    # Pattern for a gerund verb followed by optional non-stopword adjectives and one or more nouns or proper nouns
    [{"POS": "VERB", "TAG": "VBG"},
     {"POS": "ADJ", "IS_STOP": False, "OP": "*"},
     {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
    
    # Pattern for one or more adjectives (excluding stopwords) followed by one or more nouns or proper nouns
    [{"POS": "ADJ", "IS_STOP": False, "OP": "*"}, 
     {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
    
    # Pattern for one or more nouns (excluding stopwords) including proper nouns
    [{"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}]
]

# Add patterns with the same ID as we process them uniformly
matcher.add("COMBINED_PATTERN", patterns)

##########################################
## DEPRECATED: This function is not used
def extract_and_format_matches(doc):
    matches = matcher(doc)
    spans = filter_spans([doc[start:end] for _, start, end in matches])
    
    # Function to check if a token contains punctuation
    def contains_punctuation(token):
        #return any(char in string.punctuation for char in token.text)
    
        return any(char in valid_chars for char in token.text)
    # Process spans to format as per requirements, excluding PROPN if they contain punctuation
    result_str = " ".join([
        "_".join([
            token.lemma_.lower() if not contains_punctuation(token) else token.lemma_
            for token in span if not (contains_punctuation(token))
        ])
        for span in spans
    ])
    
    return result_str

############################################

def has_consecutive_hyphens(text):
    # Loop through the string and check for consecutive hyphens
    for i in range(len(text) - 1):
        if text[i] == '-' and text[i + 1] == '-':
            return True
    return False

# Extract and format words
def extract_words(doc: Doc) -> str:
    """
    @param doc: A spaCy Doc object
    @return: A string of cleaned, filtered noun phrases
    """
    
    filtered_words = []
    
    for word in doc:

        token = word.text
        
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

        if token in stopwords:
            continue

        if token.isdigit():
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

            # Add the clean word
            filtered_words.append(subword_lemmatized)
        else:
            # Add the clean word
            word_lemmatized = nlp(token)[0].lemma_
            
            if word_lemmatized in stopwords:
                continue
                
            filtered_words.append(word_lemmatized)

    # Join the valid, cleaned words
    return ' '.join(filtered_words)


# Extract and format matches
def extract_noun_phrases(doc: Doc) -> str:
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
def process_text(text: str, chunk: str) -> str:
    """
    Process text to extract noun phrases, clean and filter them according to custom rules.

    @param text: A string containing raw text
    @return: A cleaned, processed string with filtered noun chunks
    """
    doc = nlp(text)

    ## Bad encoding from OpenAlex may lead to errors in processing
    ## Better to skip these papers and do not overload with many exception handlings
    
    if chunk == 'words':
        try:
            return extract_words(doc)
        except Exception:
            return np.nan
            
    if chunk == 'phrases':
        try:
            return extract_noun_phrases(doc)
        except Exception:
            return np.nan


def plain_text_from_inverted(inverted_index):
    
    if inverted_index is None:
        return None

    positions = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))

    positions.sort()

    return ' '.join([word for index, word in positions])
