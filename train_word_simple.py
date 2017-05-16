# -*- coding: utf-8 -*-
import sys
from io import open
from math import ceil

import numpy as np

from keras.layers import Dense, Dropout, Input, Activation, Embedding, SpatialDropout1D, GRU
from keras.models import Model
from keras.optimizers import Adam

import utils
from Encoder.Encoder_Predictor import Encoder_Predictor
from Encoder.Encoder_Processor import Encoder_Processor
from Encoder.Encoder_RL_layer import Encoder_RL_Layer
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
    settings['sentence_embedding_size'] = 64
    settings['depth'] = 6
    settings['action_dim'] = 64
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.0
    settings['dropout_action'] = 0.0
    settings['dropout_emb'] = 0.0
    settings['hidden_dims'] = [64]
    settings['dense_dropout'] = 0.0
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 6
    settings['max_len'] = 128
    settings['max_features']=10000
    settings['with_sentences']=False
    settings['epochs'] = 50
    settings['random_action_prob'] = 0.3
    settings['copy_etp'] = copy_weights_encoder_to_predictor_wordbased
    settings['with_embedding'] = False
    settings['l2'] = 0.00001
    settings['epoch_mult'] = 10
    return settings

def prepare_objects(data, settings):
    with open(utils.INDEXES_FILENAME, "rt", encoding="utf8") as f:
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
    data_input = Input(shape=(settings['max_len'],))
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

    encoder = GRU(units=settings['sentence_embedding_size'],
                                recurrent_dropout=settings['dropout_U'],
                                dropout=settings['dropout_W'],
                                name='encoder')(embedding)
    layer = encoder

    for idx, hidden_dim in enumerate(settings['hidden_dims']):
        layer = Dropout(settings['dense_dropout'])(layer)
        layer = Dense(hidden_dim, name='dense_{}'.format(idx))(layer)
        layer = Activation('tanh')(layer)
    layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(inputs=data_input, outputs=output)
    optimizer = Adam()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model


def build_generator_HRNN(data, settings, indexes):
    def generator():
        walk_order = list(indexes)
        np.random.shuffle(walk_order)
        bucket = []
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
            bucket.append((sentence, label))
            if len(bucket)==settings['batch_size']:
                X, Y = build_batch(data, settings, bucket)
                bucket = []
                yield X, Y
    return generator()

def build_batch(data, settings, sentence_batch):
    X = np.zeros((settings['batch_size'], settings['max_len']))
    Y = np.zeros((settings['batch_size'], settings['num_of_classes']), dtype=np.bool)
    for i, sentence_tuple in enumerate(sentence_batch):
        for idx, word in enumerate(sentence_tuple[0]):
            if word in data['word_corpus_encode']:
                X[i][idx] = data['word_corpus_encode'][word]+1
            else:
                X[i][idx] = settings['max_features']+1
        Y[i][data['labels'].index(sentence_tuple[1])] = True
    return X, Y

def run_training_simple(data, objects, settings):
    model = objects['model']
    epoch_size = int(len(objects['train_indexes'])/(settings['epoch_mult']*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])/(1*settings['batch_size']))

    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))

    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss_total = []
        acc_total = []
        for j in range(epoch_size):
            batch = next(objects['data_gen'])
            loss1 = model.train_on_batch(batch[0], batch[1])
            loss_total.append(loss1[0])
            acc_total.append(loss1[1])

            if len(loss_total) == 0:
                avg_loss = 0
            else:
                avg_loss = np.sum(loss_total)*1.0/len(loss_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)*1.0/len(acc_total)
            sys.stdout.write("\r batch {} / {}: loss = {:.4f}, acc = {:.4f}"
                         .format(j+1, epoch_size,
                                 avg_loss, avg_acc))


        sys.stdout.write("\n")
        loss_total = []
        acc_total = []
        for i in range(val_epoch_size):
            batch = next(objects['val_gen'])
            loss = model.evaluate(batch[0], batch[1], batch_size=settings['batch_size'], verbose=0)

            loss_total.append(loss[0])
            acc_total.append(loss[1])
            sys.stdout.write("\r Testing batch {} / {}: loss1 = {:.4f}, acc = {:.4f}"
                             .format(i+1, val_epoch_size,
                                     np.sum(loss_total)*1.0/len(loss_total),
                                     np.sum(acc_total)*1.0/len(acc_total)))



###############################################################


def train(filename):
    settings = init_settings()
    settings['with_sentences']=True
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    #load(objects, filename)
    sys.stdout.write('Compiling model\n')
    run_training_simple(data, objects, settings)
    #run_training_encoder_only(data, objects, settings)
    #run_training_RL_only(data, objects, settings)
    #save(objects, filename)


if __name__=="__main__":
    train("model")