# -*- coding: utf-8 -*-
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer



from keras.initializers import glorot_uniform, orthogonal, zeros
import theano as T
import theano.tensor as TS
from theano.printing import Print

class Encoder(Layer):
    def __init__(self, input_dim, units, units_ep, l2, dropout_w, dropout_u, batch_size, **kwargs):
        self.input_dim = input_dim
        self.units = units
        self.units_ep = units_ep
        self.l2 = l2
        self.epsilon = 1e-5
        self.dropout_w = dropout_w
        self.dropout_u = dropout_u
        self.batch_size = batch_size
        super(Encoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.input_dim, 3*self.units),
                                 initializer=glorot_uniform(),
                                 regularizer=l2(self.l2),
                                 name='W')
        self.U = self.add_weight(shape=(self.units, 3*self.units),
                                 initializer=orthogonal(),
                                 regularizer=l2(self.l2),
                                 name='U')
        self.b = self.add_weight(shape=(3*self.units),
                                 initializer=zeros(),
                                 name='b')
        self.gammas = self.add_weight(shape=(2, 3*self.units,),
                                      initializer=zeros(),
                                      name='gammas')
        self.betas = self.add_weight(shape=(2, 3*self.units,),
                                     initializer=zeros(),
                                     name='betas')

        self.W_EP = self.add_weight(shape=(self.input_dim, 3*self.units_ep),
                                    trainable=False,
                                 initializer=glorot_uniform(),
                                 regularizer=l2(self.l2),
                                 name='W_EP')
        self.U_EP = self.add_weight(shape=(self.units_ep, 3*self.units_ep),
                                    trainable=False,
                                 initializer=orthogonal(),
                                 regularizer=l2(self.l2),
                                 name='U_EP')
        self.b_EP = self.add_weight(shape=(3*self.units_ep),
                                    trainable=False,
                                 initializer=zeros(),
                                 name='b_EP')
        self.gammas_EP = self.add_weight(shape=(2, 3*self.units_ep,),
                                         trainable=False,
                                      initializer=zeros(),
                                      name='gammas_EP')
        self.betas_EP = self.add_weight(shape=(2, 3*self.units_ep,),
                                        trainable=False,
                                     initializer=zeros(),
                                     name='betas_EP')
        self.W1_EP = self.add_weight(shape=(self.units_ep, 1),
                                     trainable=False,
                                  initializer=glorot_uniform(),
                                  regularizer=l2(self.l2),
                                  name='W1_EP')
        self.b1_EP = self.add_weight(shape=(1),
                                     trainable=False,
                                 initializer=zeros(),
                                 name='b1_EP')

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return [None, None]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1], self.units),
                (input_shape[0], input_shape[1])]


    def call(self, input, mask=None):
        x = input
        data_mask = mask
        if data_mask.ndim == x.ndim-1:
            data_mask = K.expand_dims(data_mask)
        assert data_mask.ndim == x.ndim
        data_mask = data_mask.dimshuffle([1,0])
        x = x.dimshuffle([1,0,2])

        initial_h = K.zeros((self.batch_size, self.units))
        initial_h_ep = K.zeros((self.batch_size, self.units_ep))
        initial_is_end = K.zeros((self.batch_size))


        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, data_mask],
                            outputs_info=[initial_h, initial_h_ep, initial_is_end])
        h = results[0]
        new_mask = results[2]

        h = h.dimshuffle([1,0,2])
        new_mask = new_mask.dimshuffle([1,0])
        return [h, new_mask]

    def horizontal_step(self, x, x_mask, h_tm1, h_ep_tm1, is_end_tm1):
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

        if 0 < self.dropout_u < 1:
            ones = K.ones((self.units_ep))
            B_U_ep = K.in_train_phase(K.dropout(ones, self.dropout_u), ones)
        else:
            B_U_ep = K.cast_to_floatx(1.)
        if 0 < self.dropout_w < 1:
            ones = K.ones((self.units_ep))
            B_W_ep = K.in_train_phase(K.dropout(ones, self.dropout_w), ones)
        else:
            B_W_ep = K.cast_to_floatx(1.)

        is_end_tm1_for_h = K.expand_dims(is_end_tm1)
        is_end_tm1_for_h = K.repeat_elements(is_end_tm1_for_h, self.units, 1)
        h_tm1 = K.switch(is_end_tm1_for_h, 0, h_tm1)

        h = self.gru_step(x, h_tm1, B_W, B_U, self.W, self.U, self.b, self.gammas, self.betas, self.units)
        x_mask_for_h = K.expand_dims(x_mask)
        x_mask_for_h = K.repeat_elements(x_mask_for_h, self.units, 1)
        h = K.switch(x_mask_for_h, h, h_tm1)

        h_ep = self.gru_step(x, h_ep_tm1, B_W_ep, B_U_ep, self.W_EP, self.U_EP, self.b_EP, self.gammas_EP, self.betas_EP, self.units_ep)
        x_mask_for_h_ep = K.expand_dims(x_mask)
        x_mask_for_h_ep = K.repeat_elements(x_mask_for_h_ep, self.units_ep, 1)
        h_ep = K.switch(x_mask_for_h_ep, h_ep, h_ep_tm1)
        is_end = K.sigmoid(K.dot(h_ep_tm1, self.W1_EP) + self.b1_EP)
        is_end = K.round(is_end)
        is_end = K.flatten(is_end)
        is_end = Print("is_end")(is_end)

        return h, h_ep, is_end

    def gru_step(self, x, h_tm1, B_W, B_U, W, U, b, gammas, betas, dim):
        s1 = self.ln(K.dot(x*B_W, W) + b, gammas[0], betas[0])
        s2 = self.ln(K.dot(h_tm1*B_U, U[:,:2*dim]), gammas[1,:2*dim], betas[1,:2*dim])
        s = K.hard_sigmoid(s1[:,:2*dim] + s2)
        z = s[:,:dim]
        r = s[:,dim:2*dim]
        h_ = z*h_tm1 + (1-z)*K.tanh(s1[:,2*dim:] + self.ln(K.dot(r*h_tm1*B_U, U[:,2*dim:]), gammas[1,2*dim:], betas[1,2*dim:]))
        return h_


    # Linear Normalization
    def ln(self, x, gammas, betas):
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
        x_normed = (x - m) / (std + self.epsilon)
        x_normed = gammas * x_normed + betas
        return x_normed