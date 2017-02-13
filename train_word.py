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


from HRNN_encoder import HRNN_encoder
import utils



CASES_FILENAME = "cases.txt"
QUOTES = ["'", '“', '"']




def get_data(settings):
    with open(utils.SPLITTED_SENTENCES_FILENAME, "rt") as f:
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
    return data, settings



def init_settings():
    settings = {}
    settings['word_embedding_size'] = 32
    settings['sentence_embedding_size'] = 128
    settings['depth'] = 8
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.2
    settings['hidden_dims'] = [64]
    settings['dense_dropout'] = 0.5
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 1
    settings['max_sentence_len_for_model'] = 1024
    settings['max_sentence_len_for_generator'] = 16
    settings['max_features']=15000
    settings['with_sentences']=False
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
    data_input = Input(shape=(settings['max_sentence_len_for_model'],))
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
            row = data['sentences'][idx]
            sentence = row['sentence']
            label = row['label']
            if len(walk_order) == 0:
                walk_order = list(indexes)
                np.random.shuffle(walk_order)
            if len(sentence) > settings['max_sentence_len_for_generator']:
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
    X = np.zeros((settings['batch_size'], settings['max_sentence_len_for_model']))
    Y = np.zeros((settings['batch_size'], settings['num_of_classes']), dtype=np.bool)
    for i, sentence_tuple in enumerate(sentence_batch):
        for idx, word in enumerate(sentence_tuple[0]):
            if word in data['word_corpus_encode']:
                X[i][idx] = data['word_corpus_encode'][word]+1
            else:
                X[i][idx] = settings['max_features']+1
        X[i][min(len(sentence_tuple[0]), settings['max_sentence_len_for_model']-1)] = settings['max_features']+2
        Y[i][data['labels'].index(sentence_tuple[1])] = True
    return X, Y

def run_training(data, objects):
    objects['model'].fit_generator(generator=objects['data_gen'], validation_data=objects['val_gen'], nb_val_samples=len(objects['val_indexes'])/10, samples_per_epoch=len(objects['train_indexes'])/10, nb_epoch=50, callbacks=[LearningRateScheduler(lr_scheduler)])


def lr_scheduler(epoch):
    return epoch*0.0001 + (50-epoch)*0.001

def train(weights_filename):
    settings = init_settings()
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    objects['model'].load_weights("yelpw5.h5")
    sys.stdout.write('Compiling model\n')
    run_training(data, objects)
    objects['model'].save_weights(weights_filename)


if __name__=="__main__":
    train("weights.h5")