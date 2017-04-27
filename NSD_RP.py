# -*- coding: utf-8 -*-
import sys
import random
import pandas
from math import floor, isnan
from collections import defaultdict

import utils
from io import open

HDD = "/media/ivodopyanov/fb66ccd0-b7e5-4198-ab3a-5ab906fc8443/home/ivodopynov/"

def prepare():
    sc_data = pandas.read_csv(HDD+"russianpost_cleaned2.csv", dtype=str, encoding="utf8")
    sys.stdout.write("Starting: \n")

    label_file = open(utils.LABELS_FILENAME, "wt", encoding="utf8")
    sentences_file = open(utils.SENTENCES_FILENAME, "wt", encoding="utf8")
    splitted_sentences_file = open(utils.SPLITTED_SENTENCES_FILENAME, "wt", encoding="utf8")
    char_count = defaultdict(int)
    word_count = defaultdict(int)
    data_count = 0
    for idx, row in sc_data.iterrows():
        if idx%1000 == 0:
            sys.stdout.write(u"\r{} / {}".format(idx, sc_data.shape[0]))
        description = row['desc']
        service = row['service']
        sevicecomp = row['sevicecomp']
        route = row['route']
        if description != description:
            continue
        data_count += 1
        for char in description:
            char_count[char] += 1
        words = description.split(u" ")
        for word in words:
            word_count[word] += 1
        label_file.write(u"{} {} {}\n".format(service, sevicecomp, route))
        sentences_file.write(description+" "+utils.EOS_WORD+u"\n")
        splitted_sentences_file.write(description+" "+utils.EOS_WORD+u"\n")


    word_corpus = list(word_count.keys())
    word_corpus.sort()
    char_corpus = list(char_count.keys())
    char_corpus.sort()

    indexes = list(range(data_count))
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