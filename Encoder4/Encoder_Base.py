# -*- coding: utf-8 -*-
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer
from keras import activations


from keras.initializers import glorot_uniform, orthogonal, zeros, ones
import theano.tensor as TS
from theano.printing import Print


class Encoder_Base(Layer):
    def __init__(self,  input_dim, inner_dim, hidden_dim, action_dim, depth, batch_size,
                 dropout_w, dropout_u, dropout_action, l2, **kwargs):
        '''
        Layer also uses
        * Layer Normalization
        * Bucketing (bucket size goes as input of shape (1,1)
        * Masking
        * Dropout

        :param input_dim: dimensionality of input tensor (num of chars\word embedding size)
        :param hidden_dim: dimensionality of hidden and output tensors
        :param depth: tree hierarchy depth
        '''
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.depth = depth
        self.batch_size = batch_size
        self.dropout_w = dropout_w
        self.dropout_u = dropout_u
        self.dropout_action = dropout_action
        self.supports_masking = True
        self.epsilon = 1e-5
        self.l2 = l2

        if self.dropout_w or self.dropout_u or self.dropout_action:
            self.uses_learning_phase = True

        super(Encoder_Base, self).__init__(**kwargs)


    def build(self, input_shape):
        self.W_emb = self.add_weight(shape=(self.input_dim, self.hidden_dim),
                                     initializer=glorot_uniform(),
                                     regularizer=l2(self.l2),
                                     name='W_emb_{}'.format(self.name))
        self.b_emb = self.add_weight(shape=(self.hidden_dim),
                                     initializer=zeros(),
                                     name='b_emb_{}'.format(self.name))

        self.W = self.add_weight(shape=(self.hidden_dim, 4*self.inner_dim),
                                 initializer=glorot_uniform(),
                                 regularizer=l2(self.l2),
                                 name='W_{}'.format(self.name))
        self.U = self.add_weight(shape=(self.hidden_dim, 4*self.inner_dim),
                                 initializer=orthogonal(),
                                 regularizer=l2(self.l2),
                                 name='U_{}'.format(self.name))
        self.b = self.add_weight(shape=(4*self.inner_dim),
                                 initializer=zeros(),
                                 name='b_{}'.format(self.name))

        '''self.W1 = self.add_weight(shape=(self.inner_dim, self.hidden_dim),
                                 initializer=glorot_uniform(),
                                 regularizer=l2(self.l2),
                                 name='W1_{}'.format(self.name))
        self.b1 = self.add_weight(shape=(self.hidden_dim),
                                 initializer=zeros(),
                                 name='b1_{}'.format(self.name))'''


        self.W_action_1 = self.add_weight(shape=(self.hidden_dim, self.action_dim),
                                          regularizer=l2(self.l2),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='W_action_1_{}'.format(self.name))
        self.U_action_1 = self.add_weight(shape=(self.hidden_dim, self.action_dim),
                                          regularizer=l2(self.l2),
                                          initializer=orthogonal(),
                                          trainable=False,
                                          name='U_action_1_{}'.format(self.name))
        self.V_action_1 = self.add_weight(shape=(self.hidden_dim, self.action_dim),
                                          regularizer=l2(self.l2),
                                          initializer=orthogonal(),
                                          trainable=False,
                                          name='V_action_1_{}'.format(self.name))
        self.b_action_1 = self.add_weight(shape=(self.action_dim,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='b_action_1_{}'.format(self.name))

        self.W_action_3 = self.add_weight(shape=(self.action_dim, 2),
                                          regularizer=l2(self.l2),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='W_action_3_{}'.format(self.name))
        self.b_action_3 = self.add_weight(shape=(2,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='b_action_3_{}'.format(self.name))

        self.built = True



    def final_step(self, x, x_mask, prev_has_value, h_tm1, has_value_tm1):
        if 0 < self.dropout_u < 1:
            ones = K.ones((self.inner_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_u), ones)
        else:
            B_U = K.cast_to_floatx(1.)
        if 0 < self.dropout_w < 1:
            ones = K.ones((self.inner_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_w), ones)
        else:
            B_W = K.cast_to_floatx(1.)
        if 0 < self.dropout_w < 1:
            ones = K.ones((self.hidden_dim))
            B_W1 = K.in_train_phase(K.dropout(ones, self.dropout_w), ones)
        else:
            B_W1 = K.cast_to_floatx(1.)

        #x = Print("x")(x)
        #h_tm1 = Print("h_tm1")(h_tm1)
        #has_value_tm1 = Print("has_value_tm1")(has_value_tm1)
        #x_mask = Print("x_mask")(x_mask)
        #prev_has_value = Print("prev_has_value")(prev_has_value)
        h_ = self.gru_step(x, h_tm1,B_W,B_U,B_W1,self.W, self.U, self.b, self.inner_dim)

        has_value_tm1_for_h = K.expand_dims(has_value_tm1)
        has_value_tm1_for_h = K.repeat_elements(has_value_tm1_for_h, self.hidden_dim, 1)
        h = K.switch(has_value_tm1_for_h, h_, x)

        mask_for_h = K.expand_dims(x_mask*prev_has_value)
        #mask_for_h = Print("mask_for_h")(mask_for_h)
        mask_for_h = K.repeat_elements(mask_for_h, self.hidden_dim, 1)
        h = K.switch(mask_for_h, h, h_tm1)
        has_value = K.switch(x_mask*prev_has_value, 1, has_value_tm1)
        has_value = TS.cast(has_value, "bool")

        #h = Print("h")(h)
        return h, has_value



    def gru_step(self, x, h_tm1, B_W, B_U, B_W1, W, U, b, dim):
        s = K.hard_sigmoid(K.dot(x*B_W, W[:,:3*dim]) + K.dot(h_tm1*B_U, U[:,:3*dim]) + b[:3*dim])
        z = s[:,:dim]
        r1 = s[:,dim:2*dim]
        r2 = s[:, 2*dim:3*dim]
        h__ = z*h_tm1 + (1-z)*K.tanh(K.dot(r1*x*B_W, W[:,3*dim:]) + K.dot(r2*h_tm1*B_U, U[:,3*dim:]) + b[3*dim:])
        return h__




    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'hidden_dim': self.hidden_dim,
                  'action_dim': self.action_dim,
                  'batch_size': self.batch_size,
                  'dropout_w': self.dropout_w,
                  'dropout_u': self.dropout_u,
                  'dropout_action': self.dropout_action,
                  'depth': self.depth}
        base_config = super(Encoder_Base, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
