# -*- coding: utf-8 -*-
import keras.backend as K
from keras import activations
from keras.engine import Layer

import theano.tensor as TS
from keras.initializers import glorot_uniform, orthogonal, zeros
from keras.regularizers import l2

class Encoder_RL_Layer(Layer):
    def __init__(self,hidden_dim, action_dim,
                 dropout_w, dropout_u, dropout_action, l2, **kwargs):
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.dropout_w = dropout_w
        self.dropout_u = dropout_u
        self.dropout_action = dropout_action
        self.l2 = l2
        if self.dropout_w or self.dropout_u or self.dropout_action:
            self.uses_learning_phase = True
        super(Encoder_RL_Layer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.W_action_1 = self.add_weight(shape=(self.hidden_dim, self.action_dim),
                                          initializer=glorot_uniform(),
                                          regularizer=l2(self.l2),
                                          name='W_action_1_{}'.format(self.name))
        self.U_action_1 = self.add_weight(shape=(self.hidden_dim, self.action_dim),
                                          initializer=orthogonal(),
                                          name='U_action_1_{}'.format(self.name))
        self.b_action_1 = self.add_weight(shape=(self.action_dim,),
                                          initializer=zeros(),
                                          name='b_action_2_{}'.format(self.name))

        self.W_action_3 = self.add_weight(shape=(self.action_dim, 2),
                                          initializer=glorot_uniform(),
                                          name='W_action_3_{}'.format(self.name))
        self.b_action_3 = self.add_weight(shape=(2,),
                                          initializer=zeros(),
                                          name='b_action_3_{}'.format(self.name))

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 2)

    def call(self, input, mask=None):
        x = input[0]
        h_tm1 = input[1]

        if 0 < self.dropout_u < 1:
            ones = K.ones((self.hidden_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_u), ones)
        else:
            B_U = K.cast_to_floatx(1.)
        if 0 < self.dropout_w < 1:
            ones = K.ones((self.hidden_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_w), ones)
        else:
            B_W = K.cast_to_floatx(1.)
        if 0 < self.dropout_action < 1:
            ones = K.ones((self.action_dim))
            B_action = K.in_train_phase(K.dropout(ones, self.dropout_action), ones)
        else:
            B_action = K.cast_to_floatx(1.)

        policy = activations.tanh(K.dot(x*B_W, self.W_action_1) + K.dot(h_tm1*B_U, self.U_action_1) + self.b_action_1)
        policy = K.exp(K.minimum(K.dot(policy*B_action, self.W_action_3)+self.b_action_3, 5))

        return policy

    def get_config(self):
        config = {'hidden_dim': self.hidden_dim,
                  'action_dim': self.action_dim,
                  'dropout_action': self.dropout_action,
                  'dropout_w': self.dropout_w,
                  'dropout_u': self.dropout_u}
        base_config = super(Encoder_RL_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))