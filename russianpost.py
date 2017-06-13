# -*- coding: utf-8 -*-
import sys
import random
from math import floor
from collections import defaultdict
import pandas as pd

import utils
from io import open

FOLDER = "/media/ivodopyanov/fb66ccd0-b7e5-4198-ab3a-5ab906fc8443/home/ivodopynov/"





def prepare():
    data = pd.read_csv(FOLDER+"russianpost_cleaned2.csv", dtype="str", encoding="utf8")

    char_count = defaultdict(int)
    word_count = defaultdict(int)
    label_file = open(utils.LABELS_FILENAME, "wt", encoding="utf8")
    sentences_file = open(utils.SENTENCES_FILENAME, "wt", encoding="utf8")
    splitted_sentences_file = open(utils.SPLITTED_SENTENCES_FILENAME, "wt", encoding="utf8")
    for idx, row in data.iterrows():
        if idx%1000 == 0:
            sys.stdout.write(u"\r{} / {}".format(idx, data.shape[0]))
        for char in row['desc']:
            char_count[char] += 1
        words = row['desc'].split(u" ")
        for word in words:
            word_count[word] += 1
        label_file.write(u"{} {} {}\n".format(row['service'], row['sevicecomp'], row['route']))
        sentences_file.write(row['desc']+" "+utils.EOS_WORD+u"\n")
        splitted_sentences_file.write(row['desc']+" "+utils.EOS_WORD+u"\n")

    indexes = list(range(data.shape[0]))
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


if __name__ == "__main__":
    prepare()