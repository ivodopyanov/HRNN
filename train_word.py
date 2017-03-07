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
    return data, settings



def init_settings():
    settings = {}
    settings['word_embedding_size'] = 32
    settings['sentence_embedding_size'] = 64
    settings['depth'] = 10
    settings['action_dim'] = 128
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.2
    settings['hidden_dims'] = [64]
    settings['dense_dropout'] = 0.5
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 8
    settings['max_len'] = 32
    settings['max_features']=10000
    settings['with_sentences']=False
    settings['epochs'] = 50
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
    data_input = Input(shape=(settings['max_len'],))
    bucket_size_input = Input(shape=(1,),dtype="int32")
    embedding = Embedding(input_dim=settings['max_features']+3,
                          output_dim=settings['word_embedding_size'],
                          name='emb',
                          mask_zero=True)(data_input)
    encoder = Encoder(input_dim=settings['word_embedding_size'],
                                   hidden_dim=settings['sentence_embedding_size'],
                                   depth=settings['depth'],
                                   action_dim=settings['action_dim'],
                                   batch_size = settings['batch_size'],
                                   max_len=settings['max_len'],
                                   name='encoder')([embedding, bucket_size_input])
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
    data_input = Input(shape=(settings['max_len'],))
    bucket_size_input = Input(shape=(1,),dtype="int32")
    embedding = Embedding(input_dim=settings['max_features']+3,
                          output_dim=settings['word_embedding_size'],
                          name='emb',
                          mask_zero=True)(data_input)
    encoder = Predictor(input_dim=settings['word_embedding_size'],
                                     hidden_dim=settings['sentence_embedding_size'],
                                     depth=settings['depth'],
                                     action_dim=settings['action_dim'],
                                     batch_size=settings['batch_size'],
                                     max_len=settings['max_len'],
                                     name='encoder')([embedding, bucket_size_input])
    layer = encoder[0]

    for idx, hidden_dim in enumerate(settings['hidden_dims']):
        layer = Dense(hidden_dim, name='dense_{}'.format(idx))(layer)
        layer = Activation('tanh')(layer)
        layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(input=[data_input, bucket_size_input], output=[output, encoder[1], encoder[2], encoder[3], encoder[4], encoder[5]])
    optimizer = Adam(lr=0.001, clipnorm=5)
    #model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

def build_RL_model(settings):
    x_input = Input(shape=(settings['sentence_embedding_size'],))
    h_tm1_input = Input(shape=(settings['sentence_embedding_size'],))
    layer = RL_Layer(settings['sentence_embedding_size'], settings['action_dim'], name='encoder')([x_input, h_tm1_input])
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
    X = np.zeros((settings['batch_size'], settings['max_len']))
    Y = np.zeros((settings['batch_size'], settings['num_of_classes']), dtype=np.bool)
    for i, sentence_tuple in enumerate(sentence_batch):
        for idx, word in enumerate(sentence_tuple[0]):
            if word in data['word_corpus_encode']:
                X[i][idx] = data['word_corpus_encode'][word]+1
            else:
                X[i][idx] = settings['max_features']+1
        X[i][min(len(sentence_tuple[0]), settings['max_len']-1)] = settings['max_features']+2
        Y[i][data['labels'].index(sentence_tuple[1])] = True
    return X, Y

def run_training(data, objects):
    objects['model'].fit_generator(generator=objects['data_gen'], validation_data=objects['val_gen'], nb_val_samples=len(objects['val_indexes'])/10, samples_per_epoch=len(objects['train_indexes'])/10, nb_epoch=50, callbacks=[LearningRateScheduler(lr_scheduler)])

def lr_scheduler(epoch):
    z = epoch/50
    return z*0.0001 + (1-z)*0.001




###############################################################


def run_training_RL(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    epoch_size = int(len(objects['train_indexes'])/(100*settings['batch_size']))

    for epoch in range(settings['epochs']):
        sys.stdout.write("\nEpoch {}\n".format(epoch))
        loss1_total = []
        acc_total = []
        loss2_total = []
        for i in range(epoch_size):
            batch = next(objects['data_gen'])
            loss1 = encoder.train_on_batch(batch[0], batch[1])

            predictor.get_layer('emb').W.set_value(K.get_value(encoder.get_layer('emb').W))
            predictor.get_layer('encoder').W_emb.set_value(K.get_value(encoder.get_layer('encoder').W_emb))
            predictor.get_layer('encoder').b_emb.set_value(K.get_value(encoder.get_layer('encoder').b_emb))
            predictor.get_layer('encoder').W.set_value(K.get_value(encoder.get_layer('encoder').W))
            predictor.get_layer('encoder').U.set_value(K.get_value(encoder.get_layer('encoder').U))
            predictor.get_layer('encoder').b.set_value(K.get_value(encoder.get_layer('encoder').b))
            predictor.get_layer('dense_0').W.set_value(K.get_value(encoder.get_layer('dense_0').W))
            predictor.get_layer('dense_0').b.set_value(K.get_value(encoder.get_layer('dense_0').b))
            predictor.get_layer('output').W.set_value(K.get_value(encoder.get_layer('output').W))
            predictor.get_layer('output').b.set_value(K.get_value(encoder.get_layer('output').b))

            y_pred = predictor.predict_on_batch(batch[0])

            output = y_pred[0]
            action = y_pred[1]
            action_calculated = y_pred[2]
            x = y_pred[3]
            h = y_pred[4]
            policy = y_pred[5]

            error = np.log(np.sum(output*batch[1], axis=1))
            X,Y = restore_exp(settings, x, error, h, policy, action_calculated)
            loss2 = rl_model.train_on_batch(X,Y)


            encoder.get_layer('encoder').W_action_1.set_value(K.get_value(rl_model.get_layer('encoder').W_action_1))
            encoder.get_layer('encoder').U_action_1.set_value(K.get_value(rl_model.get_layer('encoder').U_action_1))
            encoder.get_layer('encoder').b_action_1.set_value(K.get_value(rl_model.get_layer('encoder').b_action_1))
            encoder.get_layer('encoder').W_action_2.set_value(K.get_value(rl_model.get_layer('encoder').W_action_2))
            encoder.get_layer('encoder').b_action_2.set_value(K.get_value(rl_model.get_layer('encoder').b_action_2))

            predictor.get_layer('encoder').W_action_1.set_value(K.get_value(rl_model.get_layer('encoder').W_action_1))
            predictor.get_layer('encoder').U_action_1.set_value(K.get_value(rl_model.get_layer('encoder').U_action_1))
            predictor.get_layer('encoder').b_action_1.set_value(K.get_value(rl_model.get_layer('encoder').b_action_1))
            predictor.get_layer('encoder').W_action_2.set_value(K.get_value(rl_model.get_layer('encoder').W_action_2))
            predictor.get_layer('encoder').b_action_2.set_value(K.get_value(rl_model.get_layer('encoder').b_action_2))

            loss1_total.append(loss1[0])
            loss2_total.append(loss2)
            acc_total.append(loss1[1])

            sys.stdout.write("\r batch {} / {}: loss1 = {:.2f}, acc = {:.2f}, loss2 = {:.6f}"
                             .format(i,
                                     epoch_size,
                                     np.sum(loss1_total)/len(loss1_total),
                                     np.sum(acc_total)/len(acc_total),
                                     np.sum(loss2_total)/len(loss2_total)))




def restore_exp(settings, x, total_error, h, policy, fk_calculated):
    error_mult = np.repeat(np.expand_dims(total_error, axis=1), fk_calculated.shape[1], axis=1)
    error_mult = np.repeat(np.expand_dims(error_mult, axis=2), fk_calculated.shape[2], axis=2)

    chosen_action = np.greater_equal(policy[:,:,:,0], policy[:,:,:,1])
    shift_action_mask = np.ones_like(error_mult)*chosen_action
    reduce_action_mask = np.ones_like(error_mult)*(1-chosen_action)

    shift_action_policy = np.concatenate((np.expand_dims(shift_action_mask*error_mult, axis=3), np.expand_dims(policy[:,:,:,1], axis=3)), axis=3)
    shift_action_policy = np.repeat(np.expand_dims(shift_action_mask, axis=3), 2, axis=3)*shift_action_policy

    reduce_action_policy = np.concatenate((np.expand_dims(policy[:,:,:,0], axis=3), np.expand_dims(reduce_action_mask*error_mult, axis=3)), axis=3)
    reduce_action_policy = np.repeat(np.expand_dims(reduce_action_mask, axis=3), 2, axis=3)*reduce_action_policy

    new_policy = shift_action_policy + reduce_action_policy

    decision_performed = np.where(fk_calculated == 1)
    x_value_input = x[decision_performed]
    h_value_input = h[decision_performed]
    policy_output = new_policy[decision_performed]


    return [x_value_input, h_value_input], policy_output








def train(weights_filename):
    settings = init_settings()
    settings['with_sentences']=True
    data, settings = get_data(settings)
    objects = prepare_objects(data, settings)
    #objects['model'].load_weights("rl.h5")
    sys.stdout.write('Compiling model\n')
    #run_training(data, objects)
    run_training_RL(data, objects, settings)
    objects['model'].save_weights(weights_filename)


if __name__=="__main__":
    train("weights.h5")





