import os
import numpy as np
import sys
import csv
import random
from math import ceil, floor


from keras.models import Model
from keras.layers import Dense, Dropout, Input, Masking, Activation
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


from HRNN_encoder import HRNN_encoder
from utils import load_dictionary, load_labels, load_char_corpus


CASES_FILENAME = "cases.txt"
QUOTES = ["'", '“', '"']






def get_data():
    dictionary = load_dictionary()
    labels = load_labels()
    char_corpus_encode, char_corpus_decode, char_count = load_char_corpus(1e-5)
    return {'dict': dictionary,
            'labels': labels,
            'char_corpus_encode': char_corpus_encode,
            'char_corpus_decode': char_corpus_decode,
            'char_count': char_count}



def init_settings():
    settings = {}
    settings['sentence_embedding_size'] = 128
    settings['depth'] = 16
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.2
    settings['hidden_dims'] = [128]
    settings['dense_dropout'] = 0.5
    settings['num_of_classes'] = 5
    settings['bucket_size_step'] = 16
    settings['batch_size'] = 64
    settings['max_sentence_len'] = 256
    return settings


def prepare_objects(data, settings):
    sys.stdout.write('sentences count: '+str(len(data['labels']))+'\n')
    indexes = list(range(0,len(data['labels'])))
    random.shuffle(indexes)

    train_segment = int(len(indexes)*0.8)
    train_indexes = indexes[:train_segment]
    val_indexes = indexes[train_segment:]

    model = build_model(data, settings)
    data_gen = build_generator_HRNN(data, settings, train_indexes)
    val_gen = build_generator_HRNN(data, settings, val_indexes)

    return {'model': model, 'data_gen': data_gen, 'val_gen': val_gen, 'train_indexes': train_indexes, 'val_indexes': val_indexes}

def build_model(data, settings):
    sys.stdout.write('building model\n')
    data_input = Input(shape=(settings['max_sentence_len'], data['char_count']))
    bucket_size_input = Input(shape=(1,),dtype="int32")
    masking = Masking()(data_input)
    encoder = HRNN_encoder(input_dim=data['char_count'],
                              hidden_dim=settings['sentence_embedding_size'],
                              depth=settings['depth'],
                              dropout_W = settings['dropout_W'],
                              dropout_U = settings['dropout_U'],
                              name='encoder')([masking, bucket_size_input])
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
            idx = walk_order.pop()
            sentence = data['dict'][idx]
            label = data['labels'][idx]
            bucket_size = ceil(len(sentence) / settings['bucket_size_step'])*settings['bucket_size_step']
            if bucket_size not in buckets:
                buckets[bucket_size] = []
            buckets[bucket_size].append((sentence, label))
            if len(buckets[bucket_size])==settings['batch_size']:
                X, Y = build_batch(data, settings, buckets[bucket_size])
                batch_sentences = buckets[bucket_size]
                buckets[bucket_size] = []

                bucket_size_input = np.zeros((settings['batch_size'],1), dtype=int)
                bucket_size_input[0][0]=bucket_size
                yield [X, bucket_size_input], Y
            if len(walk_order) == 0:
                walk_order = list(indexes)
                np.random.shuffle(walk_order)
    return generator()

def build_batch(data, settings, sentence_batch):
    X = np.zeros((settings['batch_size'], settings['max_sentence_len'], data['char_count']))
    Y = np.zeros((settings['batch_size'], settings['num_of_classes']), dtype=np.bool)
    for i, sentence_tuple in enumerate(sentence_batch):
        result_ch_pos = 0
        sentence = sentence_tuple[0]
        label = floor(sentence_tuple[1] * settings['num_of_classes'])
        if label == 5:
            label = 4
        for ch_pos in range(len(sentence)):
            if sentence[ch_pos] in data['char_corpus_encode']:
                X[i][result_ch_pos][data['char_corpus_encode'][sentence[ch_pos]]] = True
            else:
                X[i][result_ch_pos][data['char_count']-3] = True
            result_ch_pos += 1
            if result_ch_pos == settings['max_sentence_len']-2:
                break
        X[i][result_ch_pos][data['char_count']-1] = True
        Y[i][label] = True
    return X, Y

def run_training(objects):
    objects['model'].fit_generator(generator=objects['data_gen'],
                                   validation_data=objects['val_gen'],
                                   nb_val_samples=len(objects['val_indexes']),
                                   samples_per_epoch=len(objects['train_indexes']),
                                   nb_epoch=100,
                                   callbacks=[ReduceLROnPlateau()])


def train(weights_filename):
    settings = init_settings()
    data = get_data()
    objects = prepare_objects(data, settings)
    sys.stdout.write('Compiling model\n')
    run_training(objects)
    objects['model'].save_weights(weights_filename)


if __name__=="__main__":
    train("weights.h5")