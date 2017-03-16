import os
import numpy as np
import sys
import csv
import random
from math import ceil, floor


from keras.models import Model
from keras.layers import Dense, Dropout, Input, Masking, Activation, Embedding
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import keras.backend as K

from Encoder import Encoder
from Predictor import Predictor
from RL_layer import RL_Layer
import utils
from train_utils import run_training2, copy_weights_encoder_to_predictor_charbased


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
    settings['sentence_embedding_size'] = 128
    settings['depth'] = 4
    settings['action_dim'] = 128
    settings['dropout_W'] = 0.5
    settings['dropout_U'] = 0.5
    settings['dropout_action'] = 0.5
    settings['hidden_dims'] = [64]
    settings['dense_dropout'] = 0.5
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 6
    settings['max_len'] = 256
    settings['max_features']=10000
    settings['with_sentences']=False
    settings['epochs'] = 50
    settings['random_action_prob'] = 0
    settings['copy_etp'] = copy_weights_encoder_to_predictor_charbased
    return settings

def prepare_objects(data, settings):
    with open(utils.INDEXES_FILENAME, "rt") as f:
        indexes = f.read().splitlines()
    indexes = [int(index) for index in indexes]
    train_segment = int(len(indexes)*0.9)

    train_indexes = indexes[:train_segment]
    val_indexes = indexes[train_segment:]

    encoder = build_encoder(data, settings)
    predictor = build_predictor(data, settings)
    rl_model = build_RL_model(settings)
    data_gen = build_generator_HRNN(data, settings, train_indexes)
    val_gen = build_generator_HRNN(data, settings, val_indexes)
    return {'encoder': encoder,
            'predictor': predictor,
            'rl_model': rl_model,
            'data_gen': data_gen,
            'val_gen': val_gen,
            'train_indexes': train_indexes,
            'val_indexes': val_indexes}

def build_encoder(data, settings):
    sys.stdout.write('Building model\n')
    data_input = Input(shape=(settings['max_len'],data['char_count']))
    bucket_size_input = Input(shape=(1,),dtype="int32")
    masking = Masking()(data_input)
    encoder = Encoder(input_dim=data['char_count'],
                                   hidden_dim=settings['sentence_embedding_size'],
                                   depth=settings['depth'],
                                   action_dim=settings['action_dim'],
                                   batch_size = settings['batch_size'],
                                   max_len=settings['max_len'],
                                   dropout_u=settings['dropout_U'],
                                   dropout_w=settings['dropout_W'],
                                   dropout_action=settings['dropout_action'],
                                   name='encoder')([masking, bucket_size_input])
    layer = encoder

    for idx, hidden_dim in enumerate(settings['hidden_dims']):
        layer = Dense(hidden_dim, name='dense_{}'.format(idx))(layer)
        layer = Activation('tanh')(layer)
        layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(input=[data_input, bucket_size_input], output=output)
    optimizer = Adam(lr=0.001, clipnorm=5)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

def build_predictor(data, settings):
    sys.stdout.write('Building model\n')
    data_input = Input(shape=(settings['max_len'],data['char_count']))
    bucket_size_input = Input(shape=(1,),dtype="int32")
    masking = Masking()(data_input)
    encoder = Predictor(input_dim=data['char_count'],
                                     hidden_dim=settings['sentence_embedding_size'],
                                     depth=settings['depth'],
                                     action_dim=settings['action_dim'],
                                     batch_size=settings['batch_size'],
                                     max_len=settings['max_len'],
                                     random_action_prob=settings['random_action_prob'],
                                     dropout_u=settings['dropout_U'],
                                     dropout_w=settings['dropout_W'],
                                     dropout_action=settings['dropout_action'],
                                     name='encoder')([masking, bucket_size_input])
    layer = encoder[0]

    for idx, hidden_dim in enumerate(settings['hidden_dims']):
        layer = Dense(hidden_dim, name='dense_{}'.format(idx))(layer)
        layer = Activation('tanh')(layer)
        layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(input=[data_input, bucket_size_input], output=[output, encoder[1], encoder[2], encoder[3], encoder[4], encoder[5], encoder[6]])
    optimizer = Adam(lr=0.001, clipnorm=5)
    return model

def build_RL_model(settings):
    x_input = Input(shape=(settings['sentence_embedding_size'],))
    h_tm1_input = Input(shape=(settings['sentence_embedding_size'],))
    layer = RL_Layer(settings['sentence_embedding_size'], settings['action_dim'], dropout_action=settings['dropout_action'], name='encoder')([x_input, h_tm1_input])
    model = Model(input=[x_input, h_tm1_input], output=layer)
    model.compile(loss='mse', optimizer='adam')
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
            if len(walk_order) == 0:
                walk_order = list(indexes)
                np.random.shuffle(walk_order)
            if len(sentence) > settings['max_len']:
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
    return generator()

def build_batch(data, settings, sentence_batch):
    X = np.zeros((settings['batch_size'], settings['max_len'], data['char_count']))
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
            if result_ch_pos == settings['max_len']-2:
                break
        X[i][result_ch_pos][data['char_count']-1] = True
        Y[i][data['labels'].index(sentence_tuple[1])] = True
    return X, Y

###############################################################

def train(filename):
    settings = init_settings()
    settings['with_sentences']=True
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    #objects['model'].load_weights("rl.h5")
    sys.stdout.write('Compiling model\n')
    #run_training(data, objects)
    run_training2(data, objects, settings)


if __name__=="__main__":
    train("model")





