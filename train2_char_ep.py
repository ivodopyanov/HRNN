import sys
import numpy as np
from random import shuffle, randint


from keras.layers import Dense, Input, GRU, RepeatVector, Masking, Activation
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from Encoder2.End_Predictor import EndPredictor
from Encoder2.Encoder import Encoder
from Encoder2.Unmask import Unmask

import utils
from train2_char_config import init_settings


def get_data(settings):
    word_corpus_encode, word_corpus_decode, cnt, mean = utils.load_word_corpus(settings['max_features'])
    char_corpus_encode, char_corpus_decode, char_count = utils.load_char_corpus(1e-5)

    data = {'words': word_corpus_decode,
            'word_freq': cnt,
            'word_freq_base': mean,
            'char_corpus_encode': char_corpus_encode,
            'char_corpus_decode': char_corpus_decode,
            'char_count': char_count}
    return data

def build_generator(data, settings, indexes):
    def generator():
        walk_order = list(indexes)
        np.random.shuffle(walk_order)
        bucket = []
        while True:
            idx = walk_order.pop()-1
            word = data['words'][idx]
            if len(walk_order) == 0:
                walk_order = list(indexes)
                np.random.shuffle(walk_order)
            bucket.append(word)
            if len(bucket)==settings['batch_size']:
                X, Y, sample_weights = build_batch(data, settings, bucket)
                bucket = []
                yield [X, Y, sample_weights]
    return generator()

def build_model_end_detector(data, settings):
    data_input = Input(shape=(settings['max_len'],data['char_count']))
    masking = Masking()(data_input)
    end_predictor = EndPredictor(return_sequences=True,
                                 input_dim=data['char_count'],
                                 units=settings['char_units_ep'],
                                 l2=settings['l2'],
                                 dropout_u=settings['dropout_u'],
                                 dropout_w=settings['dropout_w'],
                                 batch_size=settings['batch_size'])(masking)
    model = Model(inputs=data_input, outputs=end_predictor)
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_objects(data, settings):
    indexes = range(len(data['words']))
    shuffle(indexes)
    train_segment = int(len(indexes)*0.9)

    train_indexes = indexes[:train_segment]
    val_indexes = indexes[train_segment:]

    end_detector_model = build_model_end_detector(data, settings)
    data_gen = build_generator(data, settings, train_indexes)
    val_gen = build_generator(data, settings, val_indexes)
    return {'end_detector_model': end_detector_model,
            'data_gen': data_gen,
            'val_gen': val_gen,
            'train_indexes': train_indexes,
            'val_indexes': val_indexes}

def build_batch(data, settings, words):
    X = np.zeros((settings['batch_size'], settings['max_len'], data['char_count']))
    Y = np.zeros((settings['batch_size'], 1))
    sample_weights = np.zeros((settings['batch_size']))
    for i, word in enumerate(words):
        full_word = randint(0,1)
        if full_word == 1:
            word_length = len(word)
        else:
            word_length = randint(0, len(word)-1)
        Y[i][0]= full_word
        sample_weights[i] = data['word_freq'][word] / data['word_freq_base']
        result_ch_pos = 0
        for ch_pos in range(word_length):
            if word[ch_pos] in data['char_corpus_encode']:
                X[i][result_ch_pos][data['char_corpus_encode'][word[ch_pos]]] = True
            else:
                X[i][result_ch_pos][data['char_count']-3] = True
            result_ch_pos += 1
            if result_ch_pos == settings['max_len']-2:
                break
    return X, Y, sample_weights

def run_training_end_detector(data, objects, settings):
    model = objects['end_detector_model']
    epoch_size = int(len(objects['train_indexes'])*1.0/(settings['epoch_mult']*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])*1.0/(1*settings['batch_size']))
    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))
    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss_total = []
        for j in range(epoch_size):
            X, Y, sample_weights = next(objects['data_gen'])
            loss = model.train_on_batch(X, Y, sample_weight=sample_weights)
            loss_total.append(loss)
            if len(loss_total) == 0:
                avg_loss = 0
            else:
                avg_loss = np.sum(loss_total)*1.0/len(loss_total)

            sys.stdout.write("\rTraining batch {} / {}: loss = {:.4f}"
                         .format(j+1, epoch_size, avg_loss))
        loss_total = []
        sys.stdout.write("\n")
        for i in range(val_epoch_size):
            X, Y, sample_weights = next(objects['val_gen'])
            loss = model.evaluate(X, Y, batch_size=settings['batch_size'], verbose=0, sample_weight=sample_weights)
            loss_total.append(loss)
            if len(loss_total) == 0:
                avg_loss = 0
            else:
                avg_loss = np.sum(loss_total)*1.0/len(loss_total)

            sys.stdout.write("\rTesting batch {} / {}: loss = {:.4f}"
                         .format(i+1, val_epoch_size, avg_loss))


def train():
    settings = init_settings()
    data = get_data(settings)
    objects = prepare_objects(data, settings)
    run_training_end_detector(data, objects, settings)
    objects['end_detector_model'].save_weights("train2char_end_detector.h5")

def test():
    settings = init_settings()
    data = get_data(settings)
    objects = prepare_objects(data, settings)
    objects['end_detector_model'].load_weights('train2char_end_detector.h5')
    while True:
        s = raw_input("Print char:")
        X, Y_true, sample_weights = build_batch(data, settings, [s])
        Y_pred = objects['end_detector_model'].predict_on_batch(X)
        pass



if __name__=="__main__":
    train()