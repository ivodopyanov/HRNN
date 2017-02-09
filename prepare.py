import sys
from utils import load_dictionary
from collections import defaultdict

from utils import CHAR_COUNTS_FILENAME, CHAR_CORPUS_FILENAME, WORD_CORPUS_FILENAME, WORD_COUNTS_FILENAME

def main():
    data = load_dictionary()
    char_count = defaultdict(int)
    word_count = defaultdict(int)
    sys.stdout.write("Calculating char and word corpuses:\n")
    for idx, sentence in enumerate(data.values()):
        if idx%1000 == 0:
            sys.stdout.write("\r{} / {}".format(idx, len(data)))
        for char in sentence:
            char_count[char] += 1
        words = sentence.split(" ")
        for word in words:
            word_count[word] += 1
    word_corpus = list(word_count.keys())
    word_corpus.sort()
    char_corpus = list(char_count.keys())
    char_corpus.sort()


    char_corpus_file = open(CHAR_CORPUS_FILENAME, "wt")
    char_counts_file = open(CHAR_COUNTS_FILENAME, "wt")
    word_corpus_file = open(WORD_CORPUS_FILENAME, "wt")
    word_counts_file = open(WORD_COUNTS_FILENAME, "wt")
    for char in char_corpus:
        char_corpus_file.write(char)
        char_counts_file.write(str(char_count[char])+"\n")
    for word in word_corpus:
        word_corpus_file.write(word+"\n")
        word_counts_file.write(str(word_count[word])+"\n")
    char_corpus_file.close()
    char_counts_file.close()
    word_corpus_file.close()
    word_counts_file.close()



if __name__ == "__main__":
    main()