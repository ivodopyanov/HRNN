# -*- coding: utf-8 -*-
import sys
import os
from io import open
from collections import Counter

from nltk.tokenize import word_tokenize

SOURCE_FILENAME = "yelp_academic_dataset_review.json"
LABELS_FILENAME = "labels.txt"
SENTENCES_FILENAME = "sentences.txt"
SPLITTED_SENTENCES_FILENAME = "splitted_sentences.txt"
CHAR_CORPUS_FILENAME = "chars.txt"
CHAR_COUNTS_FILENAME = "char_counts.txt"
WORD_CORPUS_FILENAME = "words.txt"
WORD_COUNTS_FILENAME = "word_counts.txt"
INDEXES_FILENAME = "indexes.txt"
EOS_WORD = u"!@#EOS#@!"
QUOTES = [u"'", u'“', u'"']


def load_char_corpus(freq_limit):
    if not os.path.isfile(CHAR_CORPUS_FILENAME):
        return {}, list(), 0
    with open(CHAR_CORPUS_FILENAME, "rt", encoding="utf8") as f:
        all_chars = f.read()
    with open(CHAR_COUNTS_FILENAME, "rt") as f:
        char_counts = f.readlines()
    char_counts = [int(s.strip()) for s in char_counts]
    total_char_count = sum(char_counts)
    char_freqs = [(char_counts[i]*1.0/total_char_count, all_chars[i]) for i in range(len(all_chars))]
    char_corpus_decode = [char_freq[1] for char_freq in char_freqs if char_freq[0] > freq_limit]
    char_corpus_decode.sort()
    char_corpus_encode = {}
    for pos in range(len(char_corpus_decode)):
        char_corpus_encode[char_corpus_decode[pos]] = pos
    #+1 - место для неизвестного символа
    #+2 - место для спецсимвола конца слова
    #+3 - место для спецсимвола конца строки
    return char_corpus_encode, char_corpus_decode, len(char_corpus_decode) + 3

def load_word_corpus(max_features):
    with open(WORD_CORPUS_FILENAME, "rt", encoding="utf8") as f:
        all_words = f.read().split("\n")[:-1]
    with open(WORD_COUNTS_FILENAME, "rt") as f:
        word_counts = f.readlines()
    word_counts = [int(s.strip()) for s in word_counts]
    cnt = Counter(dict(zip(all_words, word_counts)))
    word_corpus_decode = []
    word_corpus_encode = {}
    mean = 0
    for idx, w in enumerate(cnt.most_common(max_features)):
        word_corpus_decode.append(w[0])
        word_corpus_encode[w[0]] = idx
        mean += w[1]
    mean = mean*1.0 / max_features
    return word_corpus_encode, word_corpus_decode, cnt, mean


def split_sentence_to_words(sentence):
    sentence = sentence_cleaning(sentence)
    words = word_tokenize(sentence)
    return words

def sentence_cleaning(sentence):
    sentence = sentence.replace("\n", " ")
    sentence = sentence.lower()
    return sentence

def strip_trailing_quotes(str):
    while True:
        has_quote = False
        for quote in QUOTES:
            if str.startswith(quote):
                str = str[1:]
                has_quote=True
        if not has_quote:
            break
    while True:
        has_quote = False
        for quote in QUOTES:
            if str.endswith(quote):
                str = str[:-1]
                has_quote=True
        if not has_quote:
            break
    return str


def save(objects, filename):
    objects['encoder'].save_weights("encoder_{}.h5".format(filename))
    objects['predictor'].save_weights("predictor_{}.h5".format(filename))
    objects['rl_model'].save_weights("rl_model_{}.h5".format(filename))

def load(objects, filename):
    objects['encoder'].load_weights("encoder_{}.h5".format(filename))
    objects['predictor'].load_weights("predictor_{}.h5".format(filename))
    objects['rl_model'].load_weights("rl_model_{}.h5".format(filename))