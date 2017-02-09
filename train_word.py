import os
import numpy as np
import sys
import csv
import random
from math import ceil, floor


from keras.models import Model
from keras.layers import Dense, Dropout, Input, Masking, Activation, Embedding
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


from HRNN_encoder import HRNN_encoder
from utils import load_dictionary, load_labels, load_word_corpus, load_indexes, load_sentences


CASES_FILENAME = "cases.txt"
QUOTES = ["'", '“', '"']






def get_data(settings):
    dictionary = load_dictionary()
    indexes = load_indexes()
    labels = load_labels()
    sentences = load_sentences()
    splitted_sentences = [[]]*len(sentences)
    sentence_labels = [0]*len(sentences)
    sys.stdout.write("Filtering root labels:")
    for phrase_idx, phrase in dictionary.items():
        if phrase_idx % 1000 == 0:
            sys.stdout.write("\rFiltering root labels: {}".format(phrase_idx))
        try:
            sentence_index = sentences.index(phrase)
            sentence_labels[sentence_index] = labels[phrase_idx]
            splitted_sentences[sentence_index] = phrase.split(" ")
        except ValueError:
            pass
    sys.stdout.write("\n")
    word_corpus_encode, word_corpus_decode = load_word_corpus(settings['max_features'])
    return {'dict': dictionary,
            'labels': labels,
            'word_corpus_encode': word_corpus_encode,
            'word_corpus_decode': word_corpus_decode,
            'indexes': indexes,
            'sentences': splitted_sentences,
            'sentence_labels': sentence_labels}



def init_settings():
    settings = {}
    settings['word_embedding_size'] = 32
    settings['sentence_embedding_size'] = 128
    settings['depth'] = 16
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.2
    settings['hidden_dims'] = [64]
    settings['dense_dropout'] = 0.5
    settings['num_of_classes'] = 5
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 32
    settings['max_sentence_len'] = 64
    settings['max_features']=10000
    settings['with_sentences']=False
    return settings


def prepare_objects(data, settings):
    model = build_model(data, settings)
    data_gen = build_generator_HRNN(data, settings, data['indexes'][0])
    test_gen = build_generator_HRNN(data, settings, data['indexes'][2])
    return {'model': model,
            'data_gen': data_gen,
            'test_gen': test_gen}

def build_model(data, settings):
    sys.stdout.write('building model\n')
    data_input = Input(shape=(settings['max_sentence_len'],))
    bucket_size_input = Input(shape=(1,),dtype="int32")
    embedding = Embedding(input_dim=settings['max_features']+3,
                          output_dim=settings['word_embedding_size'],
                          mask_zero=True)(data_input)
    encoder = HRNN_encoder(input_dim=settings['word_embedding_size'],
                              hidden_dim=settings['sentence_embedding_size'],
                              depth=settings['depth'],
                              dropout_W = settings['dropout_W'],
                              dropout_U = settings['dropout_U'],
                              name='encoder')([embedding, bucket_size_input])
    layer = encoder

    for hidden_dim in settings['hidden_dims']:
        layer = Dense(hidden_dim)(layer)
        layer = Activation('tanh')(layer)
        layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(input=[data_input, bucket_size_input], output=output)
    optimizer = Adam(lr=0.001, clipnorm=5)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

def build_generator_HRNN(data, settings, indexes):
    def generator():
        walk_order = list(indexes)
        np.random.shuffle(walk_order)
        buckets = {}
        while True:
            idx = walk_order.pop()-1
            sentence = data['sentences'][idx]
            label = data['sentence_labels'][idx]
            if len(sentence) > settings['max_sentence_len']:
                continue
            bucket_size = ceil((len(sentence)+1) / settings['bucket_size_step'])*settings['bucket_size_step']
            if bucket_size not in buckets:
                buckets[bucket_size] = []
            buckets[bucket_size].append((sentence, label))
            if len(buckets[bucket_size])==settings['batch_size']:
                X, Y = build_batch(data, settings, buckets[bucket_size])
                batch_sentences = buckets[bucket_size]
                buckets[bucket_size] = []

                bucket_size_input = np.zeros((settings['batch_size'],1), dtype=int)
                bucket_size_input[0][0]=bucket_size
                if settings['with_sentences']:
                    yield [X, bucket_size_input], Y, batch_sentences
                else:
                    yield [X, bucket_size_input], Y
            if len(walk_order) == 0:
                walk_order = list(indexes)
                np.random.shuffle(walk_order)
    return generator()

def build_batch(data, settings, sentence_batch):
    X = np.zeros((settings['batch_size'], settings['max_sentence_len']))
    Y = np.zeros((settings['batch_size'], settings['num_of_classes']), dtype=np.bool)
    for i, sentence_tuple in enumerate(sentence_batch):
        for idx, word in enumerate(sentence_tuple[0]):
            if word in data['word_corpus_encode']:
                X[i][idx] = data['word_corpus_encode'][word]
            else:
                X[i][idx] = settings['max_features']+1
        X[i][len(sentence_tuple[0])] = settings['max_features']+2
        label = floor(sentence_tuple[1] * settings['num_of_classes'])
        if label == 5:
            label = 4
        Y[i][label] = True
    return X, Y

def run_training(data, objects):
    objects['model'].fit_generator(generator=objects['data_gen'], validation_data=objects['test_gen'], nb_val_samples=len(data['indexes'][2]), samples_per_epoch=len(data['indexes'][0]), nb_epoch=100, callbacks=[ReduceLROnPlateau()])


def train(weights_filename):
    settings = init_settings()
    data = get_data(settings)
    objects = prepare_objects(data, settings)
    sys.stdout.write('Compiling model\n')
    run_training(data, objects)
    objects['model'].save_weights(weights_filename)

def test():
    settings = init_settings()
    settings['with_sentences'] = True
    settings['batch_size']=1
    data = get_data(settings)
    objects = prepare_objects(data, settings)
    objects['model'].load_weights("sttw1.h5")

    indexes = list(range(0,len(data['labels'])))
    random.shuffle(indexes)

    train_segment = int(len(indexes)*0.8)
    train_indexes = indexes[:train_segment]

    data_gen = build_generator_HRNN(data, settings, train_indexes)
    batch = next(data_gen)
    for str in batch[2]:
        print(str)
    objects['model'].train_on_batch(batch[0], batch[1])

if __name__=="__main__":
    train("weights.h5")