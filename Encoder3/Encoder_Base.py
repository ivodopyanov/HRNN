# -*- coding: utf-8 -*-
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer
from keras import activations


from keras.initializers import glorot_uniform, orthogonal, zeros, ones
import theano.tensor as TS
from theano.printing import Print


class Encoder_Base(Layer):
    def __init__(self, input_dim, inner_dim, hidden_dim, action_dim, depth, batch_size,
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

        self.W = self.add_weight(shape=(self.hidden_dim, self.inner_dim),
                                 initializer=glorot_uniform(),
                                 regularizer=l2(self.l2),
                                 name='W_{}'.format(self.name))
        self.U = self.add_weight(shape=(self.hidden_dim, self.inner_dim),
                                 initializer=orthogonal(),
                                 regularizer=l2(self.l2),
                                 name='U_{}'.format(self.name))
        self.b = self.add_weight(shape=(self.inner_dim),
                                 initializer=zeros(),
                                 name='b_{}'.format(self.name))

        self.W1 = self.add_weight(shape=(self.inner_dim, self.hidden_dim),
                                 initializer=glorot_uniform(),
                                 regularizer=l2(self.l2),
                                 name='W1_{}'.format(self.name))
        self.b1 = self.add_weight(shape=(self.hidden_dim),
                                 initializer=zeros(),
                                 name='b1_{}'.format(self.name))


        self.W_action_1 = self.add_weight(shape=(self.hidden_dim, self.action_dim),
                                          regularizer=l2(self.l2),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='W_action_1_{}'.format(self.name))
        self.V_action_1 = self.add_weight(shape=(self.hidden_dim, self.action_dim),
                                          regularizer=l2(self.l2),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='V_action_1_{}'.format(self.name))
        self.U_action_1 = self.add_weight(shape=(self.hidden_dim, self.action_dim),
                                          regularizer=l2(self.l2),
                                          initializer=orthogonal(),
                                          trainable=False,
                                          name='U_action_1_{}'.format(self.name))
        self.b_action_1 = self.add_weight(shape=(self.action_dim,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='b_action_2_{}'.format(self.name))

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
