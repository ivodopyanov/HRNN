# -*- coding: utf-8 -*-
import sys
import random
from collections import defaultdict

import utils



def prepare():
    text_count = [0]*5
    label_file = open(utils.LABELS_FILENAME, "wt", encoding="utf8")
    sentences_file = open(utils.SENTENCES_FILENAME, "wt", encoding="utf8")
    splitted_sentences_file = open(utils.SPLITTED_SENTENCES_FILENAME, "wt", encoding="utf8")
    char_count = defaultdict(int)
    word_count = defaultdict(int)

    with open(utils.SOURCE_FILENAME, "rt") as f:
        while True:
            total_sum = sum(text_count)
            if total_sum == 145000*5:
                break
            if total_sum % 10000 == 0:
                sys.stdout.write("\r {} {} {} {} {}".format(text_count[0], text_count[1], text_count[2], text_count[3], text_count[4]))
            s = f.readline()
            st_index = s.index("\"stars\"")
            st_value = int(s[st_index+9])-1
            if text_count[st_value] >= 145000:
                continue
            text_index = s.index("\"text\"")
            text_start_index = text_index+9
            text_end_index = s.index("\"", text_start_index)
            text = s[text_start_index:text_end_index]

            text_count[st_value] += 1

            label_file.write(str(st_value)+"\n")
            sentences_file.write(text+" "+utils.EOS_WORD+"\n")
            cleaned_text = utils.split_sentence_to_words(text)
            splitted_sentences_file.write(" ".join(cleaned_text)+" "+utils.EOS_WORD+"\n")

            for char in text:
                char_count[char] += 1
            for word in cleaned_text:
                word_count[word] += 1

    indexes = list(range(145000*5))
    random.shuffle(indexes)

    with open(utils.INDEXES_FILENAME, "wt", encoding="utf8") as f:
        for index in indexes:
            f.write(str(index)+"\n")
    char_corpus = list(char_count.keys())
    char_corpus.sort()
    word_corpus = list(word_count.keys())
    word_corpus.sort()

    with open(utils.CHAR_CORPUS_FILENAME, "wt", encoding="utf8") as f:
        for char in char_corpus:
            f.write(char)
    with open(utils.CHAR_COUNTS_FILENAME, "wt", encoding="utf8") as f:
        for char in char_corpus:
            f.write("{}\n".format(char_count[char]))
    with open(utils.WORD_CORPUS_FILENAME, "wt", encoding="utf8") as f:
        for word in word_corpus:
            f.write(word+"\n")
    with open(utils.WORD_COUNTS_FILENAME, "wt", encoding="utf8") as f:
        for word in word_corpus:
            f.write("{}\n".format(word_count[word]))

    label_file.close()
    sentences_file.close()
    splitted_sentences_file.close()

if __name__ == "__main__":
    prepare()