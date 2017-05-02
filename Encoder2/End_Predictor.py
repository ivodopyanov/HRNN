# -*- coding: utf-8 -*-
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer



from keras.initializers import glorot_uniform, orthogonal, zeros
import theano as T
import theano.tensor as TS
from theano.printing import Print

class EndPredictor(Layer):
    def __init__(self, return_sequences, input_dim, units, l2, dropout_w, dropout_u, batch_size, **kwargs):
        self.return_sequences = return_sequences
        self.input_dim = input_dim
        self.units = units
        self.l2 = l2
        self.epsilon = 1e-5
        self.dropout_w = dropout_w
        self.dropout_u = dropout_u
        self.batch_size = batch_size
        super(EndPredictor, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight((self.input_dim, 3*self.units),
                                 initializer=glorot_uniform(),
                                 regularizer=l2(self.l2),
                                 name='W_{}'.format(self.name))
        self.U = self.add_weight((self.units, 3*self.units),
                                 initializer=orthogonal(),
                                 regularizer=l2(self.l2),
                                 name='U_{}'.format(self.name))
        self.b = self.add_weight((3*self.units),
                                 initializer=zeros(),
                                 name='b_{}'.format(self.name))
        self.gammas = self.add_weight((2, 3*self.units,),
                                      initializer=zeros(),
                                      name='gammas_{}'.format(self.name))
        self.betas = self.add_weight((2, 3*self.units,),
                                     initializer=zeros(),
                                     name='betas_{}'.format(self.name))

        self.W1 = self.add_weight((self.units, 1),
                                  initializer=glorot_uniform(),
                                  regularizer=l2(self.l2),
                                  name='W1_{}'.format(self.name))
        self.b1 = self.add_weight((1),
                                 initializer=zeros(),
                                 name='b1_{}'.format(self.name))

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], 1)
        else:
            return (input_shape[0], input_shape[1], 1)


    def call(self, input, mask=None):
        x = input
        data_mask = mask
        if data_mask.ndim == x.ndim-1:
            data_mask = K.expand_dims(data_mask)
        assert data_mask.ndim == x.ndim
        data_mask = data_mask.dimshuffle([1,0])
        x = x.dimshuffle([1,0,2])

        initial_h = K.zeros((self.batch_size, self.units))
        initial_result = K.zeros((self.batch_size, 1))


        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, data_mask],
                            outputs_info=[initial_h])
        results, _ = T.scan(self.final_step,
                            sequences=[results],
                            outputs_info=[initial_result])
        if self.return_sequences:
            return results[-1]
        else:
            return results

    def final_step(self, x, h_tm1):
        return K.sigmoid(K.dot(x,self.W1)+self.b1)

    def horizontal_step(self, x, x_mask, h_tm1):
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

        h = self.gru_step(x, h_tm1, B_W, B_U)
        x_mask_for_h = K.expand_dims(x_mask)
        x_mask_for_h = K.repeat_elements(x_mask_for_h, self.units, 1)
        h = K.switch(x_mask_for_h, h, h_tm1)
        return h

    def gru_step(self, x, h_tm1, B_W, B_U):
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