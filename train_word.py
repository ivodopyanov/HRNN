# -*- coding: utf-8 -*-
import sys
from io import open
from math import ceil

import numpy as np

from keras.layers import Dense, Dropout, Input, Activation, Embedding, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from theano.printing import Print

import utils
from Encoder4.Encoder_Predictor import Encoder_Predictor
from Encoder4.Encoder_Processor import Encoder_Processor
from Encoder4.Encoder_RL_layer import Encoder_RL_Layer
from train_utils import run_training2, copy_weights_encoder_to_predictor_wordbased, run_training_encoder_only, run_training_RL_only

CASES_FILENAME = "cases.txt"
QUOTES = ["'", '“', '"']
GLOVE_FILE = "glove.6B.200d.txt"

def get_data(settings):
    with open(utils.SPLITTED_SENTENCES_FILENAME, "rt", encoding="utf8") as f:
        sentences = f.read().split(utils.EOS_WORD)
    with open(utils.LABELS_FILENAME, "rt", encoding="utf8") as f:
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
        sentence = sentence.strip(" ")
        parsed_sentence = sentence.split(" ")
        result.append({'label': labels[sentence_pos], "sentence": parsed_sentence})
        labels_set.add(labels[sentence_pos])
    labels_list = list(labels_set)
    labels_list.sort()
    sys.stdout.write("\n")

    from collections import Counter
    cnt = Counter([len(l['sentence']) for l in result])

    word_corpus_encode, word_corpus_decode, cnt, mean = utils.load_word_corpus(settings['max_features'])
    settings['num_of_classes'] = len(labels_list)
    data = {'labels': labels_list,
            'sentences': result,
            'word_corpus_encode': word_corpus_encode,
            'word_corpus_decode': word_corpus_decode}
    if settings['with_embedding']:
        settings, data = update_corpus_with_glove(settings, data)
    return data, settings


def load_glove_embeddings(data):
    result = {}
    with open(GLOVE_FILE, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in data['word_corpus_encode']:
                coef = np.asarray(values[1:], dtype='float32')
                result[word] = coef
    return result

def update_corpus_with_glove(settings, data):
    glove_emb = load_glove_embeddings(data)
    emb_matrix = np.zeros((settings['max_features']+2, settings['word_embedding_size']))
    for idx, word in enumerate(data['word_corpus_decode']):
        if word in glove_emb:
            emb_matrix[idx] = glove_emb[word]
    emb_matrix = np.asarray(emb_matrix)
    data['emb_matrix']=emb_matrix
    return settings, data

def init_settings():
    settings = {}
    settings['word_embedding_size'] = 32
    settings['sentence_embedding_size'] = 32
    settings['inner_dim'] = 32
    settings['depth'] = 20
    settings['action_dim'] = 32
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.0
    settings['dropout_action'] = 0.2
    settings['dropout_emb'] = 0.0
    settings['hidden_dims'] = [128]
    settings['dense_dropout'] = 0.5
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 16
    settings['max_len'] = 64
    settings['current_max_len']=8
    settings['max_features']=30000
    settings['with_sentences']=False
    settings['epochs'] = 200
    settings['random_action_prob_max'] = 0.5
    settings['random_action_prob_min'] = 0.1
    settings['random_action_prob_decay'] = 0.9
    settings['copy_etp'] = copy_weights_encoder_to_predictor_wordbased
    settings['with_embedding'] = False
    settings['l2'] = 0.00001
    settings['epoch_mult'] = 10
    settings['rl_gamma']=0.8
    return settings

def prepare_objects(data, settings):
    with open(utils.INDEXES_FILENAME, "rt", encoding="utf8") as f:
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
    data_input = Input(shape=(settings['max_len'],), dtype="int32")
    bucket_size_input = Input(shape=(1,),dtype="int32")
    if 'emb_matrix' in data:
        embedding = Embedding(input_dim=settings['max_features']+2,
                          output_dim=settings['word_embedding_size'],
                          name='emb',
                          mask_zero=True,
                          weights=[data['emb_matrix']],
                          trainable=False)(data_input)
    else:
        embedding = Embedding(input_dim=settings['max_features']+2,
                          output_dim=settings['word_embedding_size'],
                          name='emb',
                          mask_zero=True)(data_input)
    if settings['dropout_emb'] > 0:
        embedding = SpatialDropout1D(settings['dropout_emb'])(embedding)

    encoder = Encoder_Processor(input_dim=settings['word_embedding_size'],
                                inner_dim=settings['inner_dim'],
                                hidden_dim=settings['sentence_embedding_size'],
                                depth=settings['depth'],
                                action_dim=settings['action_dim'],
                                batch_size = settings['batch_size'],
                                dropout_u=settings['dropout_U'],
                                dropout_w=settings['dropout_W'],
                                dropout_action=settings['dropout_action'],
                                l2=settings['l2'],
                                name='encoder')([embedding, bucket_size_input])
    layer = encoder

    for idx, hidden_dim in enumerate(settings['hidden_dims']):
        layer = Dropout(settings['dense_dropout'])(layer)
        layer = Dense(hidden_dim, name='dense_{}'.format(idx))(layer)
        layer = Activation('tanh')(layer)
    layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(inputs=[data_input, bucket_size_input], outputs=[output])
    optimizer = Adam()

    model.compile(loss={"output": "categorical_crossentropy"}, optimizer=optimizer, metrics={"output":'accuracy'})
    return model

def build_predictor(data, settings):
    sys.stdout.write('Building model\n')
    data_input = Input(shape=(settings['max_len'],), dtype="int32")
    bucket_size_input = Input(shape=(1,),dtype="int32")
    random_action_prob = Input(shape=(1,), dtype="float32")
    if 'emb_matrix' in data:
        embedding = Embedding(input_dim=settings['max_features']+2,
                          output_dim=settings['word_embedding_size'],
                          name='emb',
                          mask_zero=True,
                          weights=[data['emb_matrix']],
                          trainable=False)(data_input)
    else:
        embedding = Embedding(input_dim=settings['max_features']+2,
                          output_dim=settings['word_embedding_size'],
                          name='emb',
                          mask_zero=True)(data_input)
    if settings['dropout_emb'] > 0:
        embedding = SpatialDropout1D(settings['dropout_emb'])(embedding)
    encoder = Encoder_Predictor(input_dim=settings['word_embedding_size'],
                                inner_dim=settings['inner_dim'],
                                hidden_dim=settings['sentence_embedding_size'],
                                depth=settings['depth'],
                                action_dim=settings['action_dim'],
                                batch_size=settings['batch_size'],
                                dropout_u=settings['dropout_U'],
                                dropout_w=settings['dropout_W'],
                                dropout_action=settings['dropout_action'],
                                l2=settings['l2'],
                                name='encoder')([embedding, bucket_size_input, random_action_prob])
    layer = encoder[0]

    for idx, hidden_dim in enumerate(settings['hidden_dims']):
        layer = Dropout(settings['dense_dropout'])(layer)
        layer = Dense(hidden_dim, name='dense_{}'.format(idx))(layer)
        layer = Activation('tanh')(layer)
    layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(inputs=[data_input, bucket_size_input, random_action_prob],
                  outputs=[output, encoder[1], encoder[2], encoder[3], encoder[4], encoder[5], encoder[6], encoder[7], encoder[8]])
    return model


def build_RL_model(settings):
    x_input = Input(shape=(settings['sentence_embedding_size'],))
    next_x_input = Input(shape=(settings['sentence_embedding_size'],))
    h_tm1_input = Input(shape=(settings['sentence_embedding_size'],))
    layer = Encoder_RL_Layer(hidden_dim=settings['sentence_embedding_size'],
                             action_dim=settings['action_dim'],
                             dropout_action=settings['dropout_action'],
                             dropout_w=settings['dropout_W'],
                             dropout_u=settings['dropout_U'],
                             l2=settings['l2'],
                             name='encoder')([x_input, next_x_input, h_tm1_input])
    model = Model(inputs=[x_input, next_x_input, h_tm1_input], outputs=[layer])
    optimizer = Adam()
    model.compile(loss='mse', optimizer=optimizer)
    return model


def build_generator_HRNN(data, settings, indexes):
    def generator():
        walk_order = list(indexes)
        #np.random.shuffle(walk_order)
        buckets = {}
        while True:
            idx = walk_order.pop()-1
            row = data['sentences'][idx]
            sentence = row['sentence']
            label = row['label']
            if len(walk_order) == 0:
                walk_order = list(indexes)
                #np.random.shuffle(walk_order)
            if len(sentence) > settings['current_max_len']:
                continue
            bucket_size = ceil((len(sentence)+1.0) / settings['bucket_size_step'])*settings['bucket_size_step']
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
    X = np.zeros((settings['batch_size'], settings['max_len']), dtype="int32")
    Y = np.zeros((settings['batch_size'], settings['num_of_classes']), dtype=np.bool)
    for i, sentence_tuple in enumerate(sentence_batch):
        for idx, word in enumerate(sentence_tuple[0]):
            if word in data['word_corpus_encode']:
                try:
                    X[i][idx] = data['word_corpus_encode'][word]+1
                except IndexError:
                    pass
            else:
                X[i][idx] = settings['max_features']+1
            Y[i][data['labels'].index(sentence_tuple[1])] = True
    return X, Y





###############################################################

def prepare_all():
    settings = init_settings()
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    return data, objects, settings

def train(filename):
    settings = init_settings()
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    #objects['encoder'].load_weights("encoder_simple.h5")
    #objects['predictor'].load_weights("predictor_simple.h5")
    #objects['rl_model'].load_weights("rl_model_simple.h5")

    #load(objects, filename)
    sys.stdout.write('Compiling model\n')
    run_training2(data, objects, settings)
    #run_training_encoder_only(data, objects, settings)
    #run_training_RL_only(data, objects, settings)
    #save(objects, filename)


if __name__=="__main__":
    train("model")