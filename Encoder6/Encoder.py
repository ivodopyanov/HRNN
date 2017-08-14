# -*- coding: utf-8 -*-
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer
from keras import activations



from keras.initializers import glorot_uniform, orthogonal, zeros, ones
import theano as T
import theano.tensor as TS
from theano.printing import Print

class Seq2Seq_Encoder(Layer):
    def __init__(self, units, word_count, l2, dropout_w, dropout_u, batch_size, kernel_size, filters, **kwargs):
        self.units = units
        self.word_count = word_count
        self.l2 = l2
        self.epsilon = 1e-5
        self.dropout_w = dropout_w
        self.dropout_u = dropout_u
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.filters = filters
        super(Seq2Seq_Encoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.word_count+1, self.units),
            initializer=glorot_uniform(),
            trainable=False,
            name='embeddings')
        self.W = self.add_weight(shape=(self.units, 3*self.units),
                                     initializer=glorot_uniform(),
                                     regularizer=l2(self.l2),
                                     trainable=False,
                                     name='W')
        self.U = self.add_weight(shape=(self.units, 3*self.units),
                                     initializer=orthogonal(),
                                     regularizer=l2(self.l2),
                                     trainable=False,
                                     name='U')
        self.b = self.add_weight(shape=(3*self.units),
                                     initializer=zeros(),
                                     trainable=False,
                                     name='b')
        self.gammas = self.add_weight(shape=(2, 3*self.units,),
                                          initializer=ones(),
                                          trainable=False,
                                          name='gammas')
        self.betas = self.add_weight(shape=(2, 3*self.units,),
                                         initializer=zeros(),
                                         trainable=False,
                                         name='betas')
        self.kernel = self.add_weight(shape=(self.kernel_size, self.units, self.filters),
                                      initializer=glorot_uniform(),
                                      name='kernel')
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (self.batch_size, self.units+self.filters)


    def call(self, input, mask=None):
        x = input[0]
        bucket_size = input[1][0][0]

        data_mask = K.not_equal(x, 0)
        data_mask = data_mask.dimshuffle([1,0])
        data_mask = data_mask[:bucket_size]

        x = K.gather(self.embeddings, x)

        x = x.dimshuffle([1,0,2])
        x = x[:bucket_size]


        initial_h = K.zeros((self.batch_size, self.units))

        results, _ = T.scan(self.horizontal_step_encoder,
                            sequences=[x, data_mask],
                            outputs_info=[initial_h])

        encoded_sentence_RNN = results[-1]


        encoded_sentence_CNN = K.conv1d(x.dimshuffle([1,0,2]), self.kernel)
        encoded_sentence_CNN = K.max(encoded_sentence_CNN, axis=1)

        return K.concatenate([encoded_sentence_RNN, encoded_sentence_CNN], axis=1)



    def horizontal_step_encoder(self, x, x_mask, h_tm1):
        h = self.gru_step(x, h_tm1)
        x_mask_for_h = K.expand_dims(x_mask)
        x_mask_for_h = K.repeat_elements(x_mask_for_h, self.units, 1)
        h = K.switch(x_mask_for_h, h, h_tm1)
        return h


    def gru_step(self, x, h_tm1):
        if 0 < self.dropout_u < 1:
            ones = K.ones((self.units))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_u), ones)
        else:
            B_U = K.cast_to_floatx(1.)
        if 0 < self.dropout_w < 1:
            ones = K.ones((self.units))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_w), ones)
        else:
            B_W = K.cast_to_floatx(1.)

        s1 = self.ln(K.dot(x*B_W, self.W) + self.b, self.gammas[0], self.betas[0])
        s2 = self.ln(K.dot(h_tm1*B_U, self.U[:,:2*self.units]), self.gammas[1,:2*self.units], self.betas[1,:2*self.units])
        s = K.hard_sigmoid(s1[:,:2*self.units] + s2)
        z = s[:,:self.units]
        r = s[:,self.units:2*self.units]
        h_ = z*h_tm1 + (1-z)*K.tanh(s1[:,2*self.units:] + self.ln(K.dot(r*h_tm1*B_U, self.U[:,2*self.units:]), self.gammas[1,2*self.units:], self.betas[1,2*self.units:]))
        return h_


    # Linear Normalization
    def ln(self, x, gammas, betas):
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
        x_normed = (x - m) / (std + self.epsilon)
        x_normed = gammas * x_normed + betas
        return x_normed