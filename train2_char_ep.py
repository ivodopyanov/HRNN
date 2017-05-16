import sys
import numpy as np
from random import shuffle, randint


from keras.layers import Dense, Input, GRU, RepeatVector, Masking, Activation
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from Encoder2.End_Predictor import EndPredictor, EndPredictorFinalStep
from Encoder2.Encoder import Encoder
from Encoder2.Unmask import Unmask

import utils
from train2_char_config import init_settings


FASTTEXT_PATH = "/media/ivodopyanov/fb66ccd0-b7e5-4198-ab3a-5ab906fc8443/home/ivodopynov/wiki.ru.vec"

def load_fasttext(settings):
    words = []
    chars = set()
    lines_count = 0
    with open(FASTTEXT_PATH, "rt") as f:
        for idx, line in enumerate(f):
            if idx==0:
                data = line.split(" ")
                lines_count = int(data[0])
                continue
            if idx % 1000 == 0:
                sys.stdout.write("\r loading fasttext {} / {}".format(idx, lines_count))
            if idx == 50000:
                break
            data = line.split(" ")
            word = data[0]
            if len(word) > settings['max_len']:
                continue
            for char in word:
                chars.add(char)
            words.append(word)
    chars = list(chars)
    chars.sort()
    chars_dict = {}
    for idx, char in enumerate(chars):
        chars_dict[char]=idx

    return words, chars_dict

def get_data(settings):
    words, chars_dict = load_fasttext(settings)
    data = {'words': words,
            'chars': chars_dict,
            'char_count': len(chars_dict)+1}
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
                X, Y, words = build_batch(data, settings, bucket)
                bucket = []
                yield [X, Y, words]
    return generator()

def build_model_end_detector(data, settings):
    data_input = Input(shape=(settings['max_len'],data['char_count']))
    masking = Masking()(data_input)
    layer = TimeDistributed(Dense(settings['char_units_ep'], activation='relu'))(masking)
    for level in range(settings['char_ep_depth']):
        layer = EndPredictor(units=settings['char_units_ep'],
                                 l2=settings['l2'],
                                 dropout_u=settings['dropout_u'],
                                 dropout_w=settings['dropout_w'],
                                 batch_size=settings['batch_size'])(layer)
    layer = EndPredictorFinalStep(return_sequences=True,
                                  batch_size=settings['batch_size'],
                                  units=settings['char_units_ep'],
                                  l2=settings['l2'])(layer)
    model = Model(inputs=data_input, outputs=layer)
    model.compile(optimizer='adam', loss='mse', metrics=['binary_accuracy'])
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
    result_words = []
    for i, word in enumerate(words):
        full_word = randint(0,1)
        if full_word == 1 or len(word) == 1:
            word_length = len(word)
        else:
            word_length = randint(1, len(word)-1)
        Y[i][0]= full_word
        result_words.append(word[0:word_length])
        result_ch_pos = 0
        for ch_pos in range(word_length):
            if word[ch_pos] in data['chars']:
                X[i][result_ch_pos][data['chars'][word[ch_pos]]] = True
            else:
                X[i][result_ch_pos][data['char_count']-1] = True
            result_ch_pos += 1
            if result_ch_pos == settings['max_len']-2:
                break
    return X, Y, result_words

def run_training_end_detector(data, objects, settings):
    model = objects['end_detector_model']
    epoch_size = int(len(objects['train_indexes'])*1.0/(settings['epoch_mult']*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])*1.0/(1*settings['batch_size']))
    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))
    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss_total = []
        acc_total = []
        for j in range(epoch_size):
            X, Y, words = next(objects['data_gen'])
            loss = model.train_on_batch(X, Y)
            loss_total.append(loss[0])
            acc_total.append(loss[1])
            if len(loss_total) == 0:
                avg_loss = 0
            else:
                avg_loss = np.sum(loss_total)*1.0/len(loss_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)*1.0/len(acc_total)

            sys.stdout.write("\rTraining batch {} / {}: loss = {:.4f}, acc = {:.4f}"
                         .format(j+1, epoch_size, avg_loss, avg_acc))
        loss_total = []
        acc_total = []
        sys.stdout.write("\n")
        for i in range(val_epoch_size):
            X, Y, words = next(objects['val_gen'])
            loss = model.evaluate(X, Y, batch_size=settings['batch_size'], verbose=0)
            loss_total.append(loss[0])
            acc_total.append(loss[1])
            if len(loss_total) == 0:
                avg_loss = 0
            else:
                avg_loss = np.sum(loss_total)*1.0/len(loss_total)
            if len(acc_total) == 0:
                avg_acc = 0
            else:
                avg_acc = np.sum(acc_total)*1.0/len(acc_total)

            sys.stdout.write("\rTesting batch {} / {}: loss = {:.4f}, acc  ={:.4f}"
                         .format(i+1, val_epoch_size, avg_loss, avg_acc))


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
        X, Y, sample_weights, words = next(objects['data_gen'])
        Y_pred = objects['end_detector_model'].predict_on_batch(X)
        pass



if __name__=="__main__":
    train()