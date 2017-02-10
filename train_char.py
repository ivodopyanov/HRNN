import os
import numpy as np
import sys
import csv
import random
from math import ceil, floor


from keras.models import Model
from keras.layers import Dense, Dropout, Input, Masking, Activation
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler


from HRNN_encoder import HRNN_encoder
import utils


CASES_FILENAME = "cases.txt"
QUOTES = ["'", '“', '"']





def get_data(settings):
    with open(utils.SENTENCES_FILENAME, "rt") as f:
        sentences = f.read().split(utils.EOS_WORD)
    with open(utils.LABELS_FILENAME, "rt") as f:
        labels = f.read().splitlines()

    sentences = sentences[:-1]

    labels_set = set()
    result = []
    print("Reading data:\n")
    for sentence_pos in range(len(sentences)):
        if sentence_pos%1000==0:
            sys.stdout.write("\r "+str(sentence_pos)+" / "+str(len(sentences)))
        sentence = utils.strip_trailing_quotes(sentences[sentence_pos])
        sentence = sentence.strip("\n")
        result.append({'label': labels[sentence_pos], "sentence": sentence})
        labels_set.add(labels[sentence_pos])
    labels_list = list(labels_set)
    labels_list.sort()
    sys.stdout.write("\n")

    char_corpus_encode, char_corpus_decode, char_count = utils.load_char_corpus(1e-5)
    settings['num_of_classes'] = len(labels_list)
    data = {'labels': labels_list,
            'sentences': result,
            'char_corpus_encode': char_corpus_encode,
            'char_corpus_decode': char_corpus_decode,
            'char_count': char_count}
    return data, settings


def init_settings():
    settings = {}
    settings['sentence_embedding_size'] = 256
    settings['depth'] = 4
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.2
    settings['hidden_dims'] = [64]
    settings['dense_dropout'] = 0.5
    settings['bucket_size_step'] = 16
    settings['batch_size'] = 64
    settings['max_sentence_len'] = 512
    return settings


def prepare_objects(data, settings):
    with open(utils.INDEXES_FILENAME, "rt") as f:
        indexes = f.read().splitlines()
    indexes = [int(index) for index in indexes]
    train_segment = int(len(indexes)*0.9)

    train_indexes = indexes[:train_segment]
    val_indexes = indexes[train_segment:]

    model = build_model(data, settings)
    data_gen = build_generator_HRNN(data, settings, train_indexes)
    val_gen = build_generator_HRNN(data, settings, val_indexes)

    return {'model': model,
            'data_gen': data_gen,
            'val_gen': val_gen,
            'train_indexes': train_indexes,
            'val_indexes': val_indexes}

def build_model(data, settings):
    sys.stdout.write('Building model\n')
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
            idx = walk_order.pop()-1
            row = data['sentences'][idx]
            sentence = row['sentence']
            label = row['label']
            if len(sentence) > settings['max_sentence_len']:
                continue
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
        for ch_pos in range(len(sentence)):
            if sentence[ch_pos] in data['char_corpus_encode']:
                X[i][result_ch_pos][data['char_corpus_encode'][sentence[ch_pos]]] = True
            else:
                X[i][result_ch_pos][data['char_count']-3] = True
            result_ch_pos += 1
            if result_ch_pos == settings['max_sentence_len']-2:
                break
        X[i][result_ch_pos][data['char_count']-1] = True
        Y[i][data['labels'].index(sentence_tuple[1])] = True
    return X, Y

def run_training(data, objects):
    objects['model'].fit_generator(generator=objects['data_gen'],
                                   validation_data=objects['val_gen'],
                                   nb_val_samples=len(objects['val_indexes'])/10,
                                   samples_per_epoch=len(objects['train_indexes'])/10,
                                   nb_epoch=50,
                                   callbacks=[LearningRateScheduler(lr_scheduler)])

def lr_scheduler(epoch):
    return epoch*0.0001 + (50-epoch)*0.001

def train(weights_filename):
    settings = init_settings()
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    sys.stdout.write('Compiling model\n')
    run_training(data, objects)
    objects['model'].save_weights(weights_filename)


if __name__=="__main__":
    train("weights.h5")