import pandas
import os
import regex
import random


import sys
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import NSD_parsing_utils
from multiprocessing import Pool

ROOT_SD_DIR = "~/SD_NLP/"
SD_COLUMNS_FILE = "servicecall_columns.txt"
SD_DATA_FILE = "servicecall.csv"


from utils import LABELS_FILENAME, SENTENCES_FILENAME, WORD_CORPUS_FILENAME, CHAR_CORPUS_FILENAME, CHAR_COUNTS_FILENAME, INDEXES_FILENAME, WORD_COUNTS_FILENAME, EOS_WORD,SPLITTED_SENTENCES_FILENAME


CASES_IGNORE = ['archival','incidentAll', 'consultation', 'konsyltac', 'serviceCall']
PRIORITIES_IGNORE = ['590101']

def prepare_data():
    sc_data = load_sc_data()
    sentence_corpus, splitted_sentence_corpus, word_corpus, word_count, char_corpus, char_count, cases = clean(sc_data['case_id'], sc_data['descriptionrtf'])
    indexes = build_indexes(len(sentence_corpus))
    save_data(cases, sentence_corpus, splitted_sentence_corpus, word_corpus, word_count,  char_corpus, char_count, indexes)

def load_sc_data():
    with open(os.path.expanduser(ROOT_SD_DIR+SD_COLUMNS_FILE)) as f:
        sc_cols = f.read().splitlines()
    sc_data = pandas.read_csv(os.path.expanduser(ROOT_SD_DIR+SD_DATA_FILE),
                                       names=sc_cols,
                                       usecols=["case_id", "descriptionrtf", "priority_id"],
                              dtype = str)
    return sc_data

def load_omnito_data():
    return pandas.read_csv(os.path.expanduser("omnito.csv"), names=['id','case_id','descriptionrtf'], dtype=str)

def clean(cases, descr):
    cnt = Counter(cases)

    cleaned_cases = []
    for (elem, count) in cnt.most_common():
        if elem not in CASES_IGNORE:
            cleaned_cases.append(elem)


    descr = descr.str.split("<b>Ответственный").str[0].str.replace("<br>$","").str.replace("<br />$","")
    sys.stdout.write("Starting: \n")
    data = [(descr.get_value(row), cases[row]) for row in range(len(descr.index)) if cases[row] in cleaned_cases and isinstance(descr.get_value(row),str)]
    pool = Pool()
    res = pool.map(process_data, data)
    sentence_corpus = list()
    splitted_sentence_corpus = list()
    cases_list = list()
    char_count = defaultdict(int)
    word_count = defaultdict(int)
    for cleaned_sentence in res:
        if len(cleaned_sentence[0]) == 0:
            continue
        for char in cleaned_sentence[0]:
            char_count[char] +=1
        for word in cleaned_sentence[1]:
            word_count[word] += 1
        sentence_corpus.append(cleaned_sentence[0])
        splitted_sentence_corpus.append(cleaned_sentence[1])
        cases_list.append(cleaned_sentence[2])


    sys.stdout.write("\n")
    word_corpus = list(word_count.keys())
    word_corpus.sort()
    char_corpus = list(char_count.keys())
    char_corpus.sort()

    return sentence_corpus, splitted_sentence_corpus, word_corpus, word_count, char_corpus, char_count, cases_list

def process_data(data):
    try:
        cleaned_sentence = NSD_parsing_utils.sentence_cleaning(data[0])
        words = NSD_parsing_utils.split_cleaned_sentence_to_words(cleaned_sentence)
        return (cleaned_sentence, words, data[1])
    except TypeError:
        return ("", [], data[1])


def build_indexes(size):
    indexes = list(range(size))
    random.shuffle(indexes)
    return indexes

def save_data(cases, sentence_corpus, splitted_sentence_corpus, word_corpus, word_count, char_corpus, char_count, indexes):
    with open(os.path.expanduser(LABELS_FILENAME), "wt", encoding="utf8") as f:
        for case in cases:
            f.write(case+"\n")
    with open(os.path.expanduser(SENTENCES_FILENAME), "wt", encoding="utf8") as f:
        for sentence in sentence_corpus:
            f.write(sentence+EOS_WORD+"\n")
    with open(os.path.expanduser(SPLITTED_SENTENCES_FILENAME), "wt", encoding="utf8") as f:
        for sentence in splitted_sentence_corpus:
            for word in sentence:
                f.write(word+" ")
            f.write(EOS_WORD+"\n")
    with open(os.path.expanduser(WORD_CORPUS_FILENAME), "wt", encoding="utf8") as f:
        for word in word_corpus:
            f.write(word+"\n")
    with open(os.path.expanduser(WORD_COUNTS_FILENAME), "wt", encoding="utf8") as f:
        for word in word_corpus:
            f.write("{}\n".format(word_count[word]))
    with open(os.path.expanduser(CHAR_CORPUS_FILENAME), "wt", encoding="utf8") as f:
        for char in char_corpus:
            f.write(char)
    with open(os.path.expanduser(CHAR_COUNTS_FILENAME), "wt", encoding="utf8") as f:
        for char in char_corpus:
            f.write("{}\n".format(char_count[char]))
    with open(os.path.expanduser(INDEXES_FILENAME), "wt", encoding="utf8") as f:
        for index in indexes:
            f.write(str(index)+"\n")

if __name__ == '__main__':
    prepare_data()