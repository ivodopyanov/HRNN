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


from FK_layer import FK_Layer
import utils
import HRNN_encoder, HRNN_encoder_RL



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
    settings['word_embedding_size'] = 64
    settings['sentence_embedding_size'] = 128
    settings['depth'] = 6
    settings['FK_dim'] = 128
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.2
    settings['hidden_dims'] = [64]
    settings['dense_dropout'] = 0.5
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 8
    settings['max_sentence_len_for_model'] = 128
    settings['max_sentence_len_for_generator'] = 64
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
    encoder = HRNN_encoder.HRNN_encoder(input_dim=settings['word_embedding_size'],
                              hidden_dim=settings['sentence_embedding_size'],
                              depth=settings['depth'],
                              FK_dim=settings['FK_dim'],
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
    z = epoch/50
    return z*0.0001 + (1-z)*0.001




###############################################################


def prepare_objects_RL(data, settings):
    with open(utils.INDEXES_FILENAME, "rt") as f:
        indexes = f.read().splitlines()
    indexes = [int(index) for index in indexes]
    train_segment = int(len(indexes)*0.9)

    train_indexes = indexes[:train_segment]
    val_indexes = indexes[train_segment:]

    model = build_model(data, settings)
    rl_model = build_RL_model(data, settings)
    fk_model = build_FK_model(settings)
    data_gen = build_generator_HRNN(data, settings, train_indexes)
    val_gen = build_generator_HRNN(data, settings, val_indexes)
    return {'model': model,
            'rl_model': rl_model,
            'fk_model': fk_model,
            'data_gen': data_gen,
            'val_gen': val_gen,
            'train_indexes': train_indexes,
            'val_indexes': val_indexes}



def build_RL_model(data, settings):
    sys.stdout.write('Building model\n')
    data_input = Input(shape=(settings['max_sentence_len_for_model'],))
    bucket_size_input = Input(shape=(1,),dtype="int32")
    embedding = Embedding(input_dim=settings['max_features']+3,
                          output_dim=settings['word_embedding_size'],
                          mask_zero=True)(data_input)
    encoder = HRNN_encoder_RL.HRNN_encoder(input_dim=settings['word_embedding_size'],
                              hidden_dim=settings['sentence_embedding_size'],
                              depth=settings['depth'],
                              FK_dim=settings['FK_dim'],
                              dropout_W = settings['dropout_W'],
                              dropout_U = settings['dropout_U'],
                              name='encoder')([embedding, bucket_size_input])
    layer = encoder[1]

    for hidden_dim in settings['hidden_dims']:
        layer = Dense(hidden_dim)(layer)
        layer = Activation('tanh')(layer)
        layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(input=[data_input, bucket_size_input], output=[output, encoder[0], encoder[2], encoder[3], encoder[4]])
    optimizer = Adam(lr=0.001, clipnorm=5)
    #model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

def run_training_RL(data, objects, settings):
    model = objects['model']
    rl_model = objects['rl_model']
    fk_model = objects['fk_model']
    epoch_size = int(len(objects['train_indexes'])/(10*settings['batch_size']))

    for epoch in range(50):
        sys.stdout.write("\nEpoch {}\n".format(epoch))
        loss1_total = []
        acc_total = []
        loss2_total = []
        for i in range(epoch_size):
            batch = next(objects['data_gen'])
            loss1 = model.train_on_batch(batch[0], batch[1])

            rl_model.layers[3].W.set_value(K.get_value(model.layers[3].W))
            rl_model.layers[3].U.set_value(K.get_value(model.layers[3].U))
            rl_model.layers[3].b.set_value(K.get_value(model.layers[3].b))
            rl_model.layers[3].gammas.set_value(K.get_value(model.layers[3].gammas))
            rl_model.layers[3].betas.set_value(K.get_value(model.layers[3].betas))

            y_pred = rl_model.predict(batch[0])

            output = y_pred[0]
            x = y_pred[1]
            h = y_pred[2]
            fk = y_pred[3]
            fk_calculated = y_pred[4]

            error = np.sum(output*batch[1], axis=1)
            X,Y = restore_exp(settings, x, error, h, fk, fk_calculated)
            loss2 = fk_model.train_on_batch(X,Y)


            model.layers[3].W_FK_1.set_value(K.get_value(fk_model.layers[3].W_FK_1))
            model.layers[3].U_FK_1.set_value(K.get_value(fk_model.layers[3].U_FK_1))
            model.layers[3].b_FK_1.set_value(K.get_value(fk_model.layers[3].b_FK_1))
            model.layers[3].action_FK_right.set_value(K.get_value(fk_model.layers[3].action_FK_right)[0,:])
            model.layers[3].action_FK_top.set_value(K.get_value(fk_model.layers[3].action_FK_top)[0,:])
            model.layers[3].W_FK_2.set_value(K.get_value(fk_model.layers[3].W_FK_2))
            model.layers[3].b_FK_2.set_value(K.get_value(fk_model.layers[3].b_FK_2))

            rl_model.layers[3].W_FK_1.set_value(K.get_value(fk_model.layers[3].W_FK_1))
            rl_model.layers[3].U_FK_1.set_value(K.get_value(fk_model.layers[3].U_FK_1))
            rl_model.layers[3].b_FK_1.set_value(K.get_value(fk_model.layers[3].b_FK_1))
            rl_model.layers[3].action_FK_right.set_value(K.get_value(fk_model.layers[3].action_FK_right)[0,:])
            rl_model.layers[3].action_FK_top.set_value(K.get_value(fk_model.layers[3].action_FK_top)[0,:])
            rl_model.layers[3].W_FK_2.set_value(K.get_value(fk_model.layers[3].W_FK_2))
            rl_model.layers[3].b_FK_2.set_value(K.get_value(fk_model.layers[3].b_FK_2))

            loss1_total.append(loss1[0])
            loss2_total.append(loss2)
            acc_total.append(loss1[1])
            if len(loss1_total) > 20:
                loss1_total.pop(0)
            if len(loss2_total) > 20:
                loss2_total.pop(0)
            if len(acc_total) > 20:
                acc_total.pop(0)


            sys.stdout.write("\r batch {} / {}: loss1 = {:.2f}, loss2 = {:.2f}, acc = {:.2f}".format(i, epoch_size, np.sum(loss1_total)/20, np.sum(acc_total)/20, np.sum(loss2_total)/20))




def restore_exp(settings, x, total_error, h, fk, fk_calculated):
    DELTA = 0.9
    left_input = []
    bottom_input = []
    action_input = []
    error_output = []
    max_steps = np.sum(fk_calculated, axis=(1,2,3))

    steps = np.zeros((fk_calculated.shape[0], fk_calculated.shape[1], fk_calculated.shape[2]))


    for j in range(h.shape[1]):
        for i in range(h.shape[2]):
            if i == 0:
                left = np.zeros((h.shape[0], h.shape[3]))
            else:
                left = h[:,j,i-1,:]
            if j == 0:
                bottom = x[:,i,:]
                z = np.zeros((bottom.shape[0], settings['sentence_embedding_size']))
                bottom = np.concatenate((bottom, z), axis=1)
            else:
                bottom = h[:, j-1, i, :]
            fk_res = fk_calculated[:, j, i, 0]
            calc = np.nonzero(fk_res)

            if i==0 and j==0:
                st = np.zeros((fk_calculated.shape[0]))
            elif i == 0:
                st = steps[:,j-1,0]
            elif j == 0:
                st = steps[:,0,i-1]
            else:
                st = np.maximum(steps[:,j,i-1], steps[:,j-1,i])
            st += fk_res
            steps[:,j,i] = st

            reward_mult = np.power(DELTA, max_steps-st)
            for batch_index in calc[0]:
                left_input.append(left[batch_index])
                bottom_input.append(bottom[batch_index])
                action_input.append(fk_res[batch_index])
                error_output.append(total_error[batch_index]*reward_mult[batch_index])
    X = [np.asarray(left_input), np.asarray(bottom_input), np.asarray(action_input)]
    Y = np.asarray(error_output)
    return (X,Y)


def build_FK_model(settings):
    x_input = Input(shape=(settings['word_embedding_size']+settings['sentence_embedding_size'],))
    h_tm1_input = Input(shape=(settings['word_embedding_size']+settings['sentence_embedding_size'],))
    action_input = Input(shape=(1,))
    layer = FK_Layer(settings['word_embedding_size'], settings['sentence_embedding_size'],settings['FK_dim'])([x_input, h_tm1_input, action_input])
    model = Model(input=[x_input, h_tm1_input, action_input], output=layer)
    model.compile(loss='mse', optimizer='adam')
    return model




def train(weights_filename):
    settings = init_settings()
    data, settings = get_data(settings)
    objects = prepare_objects_RL(data, settings)
    #objects['model'].load_weights("sttw6.h5")
    sys.stdout.write('Compiling model\n')
    #run_training(data, objects)
    run_training_RL(data, objects, settings)
    objects['model'].save_weights(weights_filename)


if __name__=="__main__":
    train("weights.h5")





