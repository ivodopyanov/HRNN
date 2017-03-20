# -*- coding: utf-8 -*-
import sys
import random
from math import floor
from collections import defaultdict

import utils
from io import open


def prepare():
    data = load_dictionary()
    labels = load_labels()
    char_count = defaultdict(int)
    word_count = defaultdict(int)
    label_file = open(utils.LABELS_FILENAME, "wt", encoding="utf8")
    sentences_file = open(utils.SENTENCES_FILENAME, "wt", encoding="utf8")
    splitted_sentences_file = open(utils.SPLITTED_SENTENCES_FILENAME, "wt", encoding="utf8")
    for idx, sentence in enumerate(data.values()):
        if idx%1000 == 0:
            sys.stdout.write(u"\r{} / {}".format(idx, len(data)))
        for char in sentence:
            char_count[char] += 1
        words = sentence.split(u" ")
        for word in words:
            word_count[word] += 1
        label_file.write(str(labels[idx])+u"\n")
        sentences_file.write(sentence+" "+utils.EOS_WORD+u"\n")
        splitted_sentences_file.write(sentence+" "+utils.EOS_WORD+u"\n")

    indexes = list(range(len(labels)))
    random.shuffle(indexes)

    with open(utils.INDEXES_FILENAME, "wt", encoding="utf8") as f:
        for index in indexes:
            f.write(str(index)+u"\n")

    word_corpus = list(word_count.keys())
    word_corpus.sort()
    char_corpus = list(char_count.keys())
    char_corpus.sort()
    char_corpus_file = open(utils.CHAR_CORPUS_FILENAME, "wt", encoding="utf8")
    char_counts_file = open(utils.CHAR_COUNTS_FILENAME, "wt", encoding="utf8")
    word_corpus_file = open(utils.WORD_CORPUS_FILENAME, "wt", encoding="utf8")
    word_counts_file = open(utils.WORD_COUNTS_FILENAME, "wt", encoding="utf8")
    for char in char_corpus:
        char_corpus_file.write(char)
        char_counts_file.write(str(char_count[char])+u"\n")
    for word in word_corpus:
        word_corpus_file.write(word+u"\n")
        word_counts_file.write(str(word_count[word])+u"\n")
    char_corpus_file.close()
    char_counts_file.close()
    word_corpus_file.close()
    word_counts_file.close()


def load_dictionary(split=False):
    result = {}
    sys.stdout.write('Loading dictionary\n')
    with open("STT/dictionary.txt", "rt", encoding="utf8") as f:
        for row in f:
            data = row.split(u"|")
            phrase = data[0]
            phrase_id = int(data[1])
            if split:
                phrase = phrase.split(u" ")
            result[phrase_id] = phrase
    return result

def load_labels():
    result = []
    with open("STT/sentiment_labels.txt", "rt", encoding="utf8") as f:
        for idx, row in enumerate(f):
            if idx == 0:
                continue
            label = float(row.split(u"|")[1])
            label = floor(label*5)
            if label == 5:
                label = 4
            result.append(label)
    return result

if __name__ == "__main__":
    prepare()