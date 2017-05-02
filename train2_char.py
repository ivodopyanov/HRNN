import sys
import numpy as np
from random import shuffle, randint
from io import open


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
    with open(utils.SENTENCES_FILENAME, "rt", encoding="utf8") as f:
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
        result.append({'label': labels[sentence_pos], "sentence": sentence})
        labels_set.add(labels[sentence_pos])
    labels_list = list(labels_set)
    labels_list.sort()
    sys.stdout.write("\n")

    char_corpus_encode, char_corpus_decode, char_count = utils.load_char_corpus(1e-5)
    settings['num_of_classes'] = len(labels_list)
    settings['hidden'] = len(labels_list)
    data = {'labels': labels_list,
            'sentences': result,
            'char_corpus_encode': char_corpus_encode,
            'char_corpus_decode': char_corpus_decode,
            'char_count': char_count}
    return data, settings

def build_generator(data, settings, indexes):
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
            bucket.append((sentence, label))
            if len(bucket)==settings['batch_size']:
                X, Y = build_batch(data, settings, bucket)
                bucket = []
                yield [X, Y]
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
                                 batch_size=settings['batch_size'],
                                 name='predictor')(masking)
    model = Model(inputs=data_input, outputs=end_predictor)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_model(data, settings, end_detection_model):
    data_input = Input(shape=(settings['max_len'],data['char_count']))
    masking = Masking()(data_input)
    char_level = Encoder(input_dim=data['char_count'],
                      units = settings['char_units'],
                      units_ep=settings['char_units_ep'],
                      l2 = settings['l2'],
                      dropout_u=settings['dropout_u'],
                      dropout_w=settings['dropout_w'],
                      batch_size=settings['batch_size'])(masking)
    unmask = Unmask()(char_level)
    word_level = GRU(units=settings['word_units'],
                     dropout=settings['dropout_w'],
                     recurrent_dropout=settings['dropout_u'])(unmask)
    hidden = Dense(units=settings['hidden'], activation='softmax')(word_level)
    model = Model(inputs=data_input, outputs=hidden)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.layers[2].W_EP=end_detection_model.layers[2].W
    model.layers[2].U_EP=end_detection_model.layers[2].U
    model.layers[2].b_EP=end_detection_model.layers[2].b
    model.layers[2].gammas_EP=end_detection_model.layers[2].gammas
    model.layers[2].betas_EP=end_detection_model.layers[2].betas
    model.layers[2].W1_EP=end_detection_model.layers[2].W1
    model.layers[2].b1_EP=end_detection_model.layers[2].b1
    return model

def prepare_objects(data, settings):
    with open(utils.INDEXES_FILENAME, "rt", encoding="utf8") as f:
        indexes = f.read().splitlines()
    indexes = [int(index) for index in indexes]
    train_segment = int(len(indexes)*0.9)

    train_indexes = indexes[:train_segment]
    val_indexes = indexes[train_segment:]

    end_detector_model = build_model_end_detector(data, settings)
    #end_detector_model.load_weights("train2char_end_detector")
    model = build_model(data, settings, end_detector_model)
    data_gen = build_generator(data, settings, train_indexes)
    val_gen = build_generator(data, settings, val_indexes)
    return {'model': model,
            'data_gen': data_gen,
            'val_gen': val_gen,
            'train_indexes': train_indexes,
            'val_indexes': val_indexes}

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
        Y[i][data['labels'].index(sentence_tuple[1])] = True
    return X, Y

def run_training_end_detector(data, objects, settings):
    model = objects['model']
    epoch_size = int(len(objects['train_indexes'])*1.0/(settings['epoch_mult']*settings['batch_size']))
    val_epoch_size = int(len(objects['val_indexes'])*1.0/(1*settings['batch_size']))
    sys.stdout.write("\nTrain epoch size = {}; val epoch size = {}".format(epoch_size, val_epoch_size))
    for epoch in range(settings['epochs']):
        sys.stdout.write("\n\nEpoch {}\n".format(epoch+1))
        loss_total = []
        for j in range(epoch_size):
            X, Y = next(objects['data_gen'])
            loss = model.train_on_batch(X, Y)
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
            X, Y = next(objects['val_gen'])
            loss = model.evaluate(X, Y, batch_size=settings['batch_size'], verbose=0)
            loss_total.append(loss)
            if len(loss_total) == 0:
                avg_loss = 0
            else:
                avg_loss = np.sum(loss_total)*1.0/len(loss_total)

            sys.stdout.write("\rTesting batch {} / {}: loss = {:.4f}"
                         .format(i+1, val_epoch_size, avg_loss))


def train():
    settings = init_settings()
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    run_training_end_detector(data, objects, settings)
    objects['model'].save_weights("train2char.h5")

def test():
    settings = init_settings()
    data = get_data(settings)
    objects = prepare_objects(data, settings)
    objects['model'].load_weights('train2char.h5')
    while True:
        s = raw_input("Print char:")
        X, Y_true = build_batch(data, settings, [s])
        Y_pred = objects['model'].predict_on_batch(X)
        pass



if __name__=="__main__":
    train()