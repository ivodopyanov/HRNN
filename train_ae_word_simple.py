# -*- coding: utf-8 -*-
import sys
from io import open
from math import ceil

import numpy as np

from keras.layers import Dense, Dropout, Input, Activation, Embedding, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam

import utils

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

    word_corpus_encode, word_corpus_decode = utils.load_word_corpus(settings['max_features'])
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
    settings['depth'] = 5
    settings['action_dim'] = 32
    settings['dropout_W'] = 0.1
    settings['dropout_U'] = 0.1
    settings['dropout_action'] = 0.1
    settings['dropout_emb'] = 0.1
    settings['hidden_dims'] = [32]
    settings['dense_dropout'] = 0.5
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 6
    settings['max_len'] = 128
    settings['max_features']=10000
    settings['with_sentences']=False
    settings['epochs'] = 50
    settings['random_action_prob'] = 0
    settings['with_embedding'] = False
    settings['l2'] = 0.00001
    settings['epoch_mult'] = 1
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

def build_processor(data, settings):
    sys.stdout.write('Building model\n')
    data_input = Input(shape=(settings['max_len'],))
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
                                   hidden_dim=settings['sentence_embedding_size'],
                                   depth=settings['depth'],
                                   action_dim=settings['action_dim'],
                                   batch_size = settings['batch_size'],
                                   max_len=settings['max_len'],
                                   dropout_u=settings['dropout_U'],
                                   dropout_w=settings['dropout_W'],
                                   dropout_action=settings['dropout_action'],
                                   l2=settings['l2'],
                                   name='encoder')([embedding, bucket_size_input])


    decoder = Decoder_Processor(hidden_dim=settings['sentence_embedding_size'],
                                output_dim=settings['word_embedding_size'],
                                depth=settings['depth'],
                                action_dim=settings['action_dim'],
                                batch_size=settings['batch_size'],
                                max_len=settings['max_len'],
                                dropout_u=settings['dropout_U'],
                                dropout_w=settings['dropout_W'],
                                dropout_action=settings['dropout_action'],
                                l2=settings['l2'],
                                name='decoder')(encoder)
    model = Model(inputs=[data_input, bucket_size_input], outputs=[decoder])
    optimizer = Adam(lr=0.001, clipnorm=5)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

def build_predictor(data, settings):
    sys.stdout.write('Building model\n')
    data_input = Input(shape=(settings['max_len'],))
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
    encoder = Encoder_Predictor(input_dim=settings['word_embedding_size'],
                                     hidden_dim=settings['sentence_embedding_size'],
                                     depth=settings['depth'],
                                     action_dim=settings['action_dim'],
                                     batch_size=settings['batch_size'],
                                     max_len=settings['max_len'],
                                     random_action_prob=settings['random_action_prob'],
                                     dropout_u=settings['dropout_U'],
                                     dropout_w=settings['dropout_W'],
                                     dropout_action=settings['dropout_action'],
                                     name='encoder')([embedding, bucket_size_input])
    decoder = Decoder_Predictor(hidden_dim=settings['sentence_embedding_size'],
                                     output_dim=settings['word_embedding_size'],
                                     depth=settings['depth'],
                                     action_dim=settings['action_dim'],
                                     batch_size=settings['batch_size'],
                                     max_len=settings['max_len'],
                                     random_action_prob=settings['random_action_prob'],
                                     dropout_u=settings['dropout_U'],
                                     dropout_w=settings['dropout_W'],
                                     dropout_action=settings['dropout_action'],
                                     name='encoder')(encoder[0])

    model = Model(inputs=[data_input, bucket_size_input], outputs=[decoder[0],
                                                                   encoder[1],
                                                                   encoder[2],
                                                                   encoder[3],
                                                                   encoder[4],
                                                                   encoder[5],
                                                                   encoder[6],
                                                                   decoder[1],
                                                                   decoder[2],
                                                                   decoder[3],
                                                                   decoder[4],
                                                                   decoder[5],
                                                                   decoder[6]])
    return model

def build_RL_model_encoder(settings):
    x_input = Input(shape=(settings['sentence_embedding_size'],))
    h_tm1_input = Input(shape=(settings['sentence_embedding_size'],))
    layer = Encoder_RL_Layer(hidden_dim=settings['sentence_embedding_size'],
                     action_dim=settings['action_dim'],
                     dropout_action=settings['dropout_action'],
                     dropout_w=settings['dropout_W'],
                     dropout_u=settings['dropout_U'],
                     l2=settings['l2'],
                     name='encoder')([x_input, h_tm1_input])
    model = Model(inputs=[x_input, h_tm1_input], outputs=[layer])
    optimizer = Adam(clipnorm=5)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def build_RL_model_decoder(settings):
    x_input = Input(shape=(settings['sentence_embedding_size'],))
    h_tm1_input = Input(shape=(settings['sentence_embedding_size'],))
    layer = Decoder_RL_Layer(hidden_dim=settings['sentence_embedding_size'],
                     action_dim=settings['action_dim'],
                     dropout_action=settings['dropout_action'],
                     dropout_w=settings['dropout_W'],
                     dropout_u=settings['dropout_U'],
                     l2=settings['l2'],
                     name='encoder')([x_input, h_tm1_input])
    model = Model(inputs=[x_input, h_tm1_input], outputs=[layer])
    optimizer = Adam(clipnorm=5)
    model.compile(loss='mse', optimizer=optimizer)
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
            bucket_size = ceil((len(sentence)+1.0) / settings['bucket_size_step'])*settings['bucket_size_step']
            if bucket_size not in buckets:
                buckets[bucket_size] = []
            buckets[bucket_size].append((sentence, label))
            if len(buckets[bucket_size])==settings['batch_size']:
                X = build_batch(data, settings, buckets[bucket_size])
                batch_sentences = buckets[bucket_size]
                buckets[bucket_size] = []

                bucket_size_input = np.zeros((settings['batch_size'],1), dtype=int)
                bucket_size_input[0][0]=bucket_size
                if settings['with_sentences']:
                    yield [X, bucket_size_input], X, batch_sentences
                else:
                    yield [X, bucket_size_input], X
    return generator()

def build_batch(data, settings, sentence_batch):
    X = np.zeros((settings['batch_size'], settings['max_len']))
    for i, sentence_tuple in enumerate(sentence_batch):
        for idx, word in enumerate(sentence_tuple[0]):
            if word in data['word_corpus_encode']:
                X[i][idx] = data['word_corpus_encode'][word]+1
            else:
                X[i][idx] = settings['max_features']+1
    return X





###############################################################


def train(filename):
    settings = init_settings()
    settings['with_sentences']=True
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    #load(objects, filename)
    sys.stdout.write('Compiling model\n')
    run_training2(data, objects, settings)
    #run_training_encoder_only(data, objects, settings)
    #run_training_RL_only(data, objects, settings)
    #save(objects, filename)


if __name__=="__main__":
    train("model")





