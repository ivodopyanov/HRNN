import sys
import os
from collections import Counter

STT_FOLDER = "stanfordSentimentTreebank/"
DICTIONARY_FILENAME = STT_FOLDER+"dictionary.txt"
LABELS_FILENAME = STT_FOLDER+"sentiment_labels.txt"
SPLIT_FILENAME = STT_FOLDER+"datasetSplit.txt"
SENTENCES_FILENAME = STT_FOLDER+"datasetSentences.txt"
CHAR_CORPUS_FILENAME = "chars.txt"
CHAR_COUNTS_FILENAME = "char_counts.txt"
WORD_CORPUS_FILENAME = "words.txt"
WORD_COUNTS_FILENAME = "word_counts.txt"

def load_dictionary():
    result = {}
    sys.stdout.write('Loading dictionary\n')
    with open(DICTIONARY_FILENAME, "rt") as f:
        for row in f:
            data = row.split("|")
            phrase = data[0]
            phrase_id = int(data[1])
            result[phrase_id] = phrase
    return result

def load_labels():
    result = []
    sys.stdout.write('Loading labels\n')
    with open(LABELS_FILENAME, "rt") as f:
        for idx, row in enumerate(f):
            if idx == 0:
                continue
            data = row.split("|")
            phrase_score = float(data[1])
            result.append(phrase_score)
    return result

def load_indexes():
    indexes = [[],[],[]]
    with open(SPLIT_FILENAME, "rt") as f:
        for idx, row in enumerate(f):
            if idx == 0:
                continue
            data = row.strip("\n").split(",")
            indexes[int(data[1])-1].append(int(data[0]))
    return indexes

def load_sentences():
    result = []
    with open(SENTENCES_FILENAME, "rt") as f:
        for idx, row in enumerate(f):
            if idx == 0:
                continue
            result.append(row.strip("\n").split("\t")[1])
    return result


def load_char_corpus(freq_limit):
    if not os.path.isfile(CHAR_CORPUS_FILENAME):
        return {}, list(), 0
    with open(CHAR_CORPUS_FILENAME, "rt", encoding="utf8") as f:
        all_chars = f.read()
    with open(CHAR_COUNTS_FILENAME, "rt") as f:
        char_counts = f.readlines()
    char_counts = [int(s.strip()) for s in char_counts]
    total_char_count = sum(char_counts)
    char_freqs = [(char_counts[i]/total_char_count, all_chars[i]) for i in range(len(all_chars))]
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
    for idx, w in enumerate(cnt.most_common(max_features)):
        word_corpus_decode.append(w[0])
        word_corpus_encode[w[0]] = idx
    return word_corpus_encode, word_corpus_decode