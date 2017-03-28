# -*- coding: utf-8 -*-
import keras.backend as K
from keras import  activations
from keras.regularizers import l2
from keras.engine import Layer


from keras.initializers import glorot_uniform, orthogonal, zeros
import numpy as np
import theano as T
import theano.tensor as TS
from theano.ifelse import ifelse
from theano.printing import Print


class Encoder_Base(Layer):
    def __init__(self, input_dim, hidden_dim, action_dim, depth, batch_size, max_len,
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
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.depth = depth
        self.batch_size = batch_size
        self.max_len = max_len
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
        self.W_emb = self.add_weight((self.input_dim, self.hidden_dim),
                                     initializer=glorot_uniform(),
                                     regularizer=l2(self.l2),
                                     name='W_emb_{}'.format(self.name))
        self.b_emb = self.add_weight((self.hidden_dim),
                                     initializer=zeros(),
                                     name='b_emb_{}'.format(self.name))

        self.W = self.add_weight((self.hidden_dim, 3*self.hidden_dim),
                                 initializer=glorot_uniform(),
                                 regularizer=l2(self.l2),
                                 name='W_{}'.format(self.name))
        self.U = self.add_weight((self.hidden_dim, 3*self.hidden_dim),
                                 initializer=orthogonal(),
                                 regularizer=l2(self.l2),
                                 name='U_{}'.format(self.name))
        self.b = self.add_weight((3*self.hidden_dim),
                                 initializer=zeros(),
                                 name='b_{}'.format(self.name))


        self.W_action_1 = self.add_weight((self.hidden_dim, self.action_dim),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='W_action_1_{}'.format(self.name))
        self.U_action_1 = self.add_weight((self.hidden_dim, self.action_dim),
                                          initializer=orthogonal(),
                                          trainable=False,
                                          name='U_action_1_{}'.format(self.name))
        self.b_action_1 = self.add_weight((self.action_dim,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='b_action_2_{}'.format(self.name))

        self.W_action_2 = self.add_weight((self.action_dim, self.action_dim),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='W_action_2_{}'.format(self.name))
        self.b_action_2 = self.add_weight((self.action_dim,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='b_action_2_{}'.format(self.name))

        self.W_action_3 = self.add_weight((self.action_dim, 2),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='W_action_3_{}'.format(self.name))
        self.b_action_3 = self.add_weight((2,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='b_action_3_{}'.format(self.name))

        self.gammas = self.add_weight((2, 3*self.hidden_dim,),
                                      initializer=zeros(),
                                      name='gammas_{}'.format(self.name))
        self.betas = self.add_weight((2, 3*self.hidden_dim,),
                                     initializer=zeros(),
                                     name='betas_{}'.format(self.name))
        self.built = True



    def final_step(self, x, x_mask, h_tm1):
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

        h = self.gru_step(x, h_tm1, B_W, B_U)
        mask_for_h = K.expand_dims(x_mask)
        mask_for_h = K.repeat_elements(mask_for_h, self.hidden_dim, 1)
        h = K.switch(mask_for_h, h, h_tm1)
        return h




    def gru_step(self, x, h_tm1, B_W, B_U):
        s1 = self.ln(K.dot(x*B_W, self.W) + self.b, self.gammas[0], self.betas[0])
        s2 = self.ln(K.dot(h_tm1*B_U, self.U[:,:2*self.hidden_dim]), self.gammas[1,:2*self.hidden_dim], self.betas[1,:2*self.hidden_dim])
        s = K.hard_sigmoid(s1[:,:2*self.hidden_dim] + s2)
        z = s[:,:self.hidden_dim]
        r = s[:,self.hidden_dim:2*self.hidden_dim]
        h_ = z*h_tm1 + (1-z)*K.tanh(s1[:,2*self.hidden_dim:] + self.ln(K.dot(r*h_tm1*B_U, self.U[:,2*self.hidden_dim:]), self.gammas[1,2*self.hidden_dim:], self.betas[1,2*self.hidden_dim:]))
        return h_

    # Linear Normalization
    def ln(self, x, gammas, betas):
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
        x_normed = (x - m) / (std + self.epsilon)
        x_normed = gammas * x_normed + betas
        return x_normed


    #Из маски 11111000000 делает маску 0000100000 (1 - на последнем элементе 1 исходной маски, остальное - 0)
    def get_current_value_mask(self, mask):
        result = 1 - mask
        result = K.concatenate([result[:, 1:], K.ones((self.batch_size, 1), dtype="bool")], axis=1)
        result = mask*result
        result = TS.cast(result, "bool")
        return result

    def get_next_value_mask(self, mask):
        result = K.concatenate([K.ones((self.batch_size, 1), dtype="bool"), mask[:,:-1]], axis=1)
        return self.get_current_value_mask(result)

    def get_prev_value_mask(self, mask):
        result = K.concatenate([mask[:, 1:], K.zeros((self.batch_size, 1), dtype="bool")], axis=1)
        return self.get_current_value_mask(result)

    def get_value_by_bitmask(self, data, mask):
        mask = mask.dimshuffle([0,1,'x'])
        mask = TS.extra_ops.repeat(mask, self.hidden_dim, axis=2)
        value = K.switch(mask, data, 0)
        value = K.sum(value, axis=1)
        return value

    def insert_tensor_at_mask(self, data, mask, tensor, bucket_size):
        # Получаем матрицу из тензора (который надо вставить), скопированных bucket_size раз
        tensor = tensor.dimshuffle((0, 'x', 1))
        tensor = TS.extra_ops.repeat(tensor, bucket_size, axis=1)
        # Получаем маску, куда надо вставить этот тензор
        mask = mask.dimshuffle([0,1,'x'])
        mask = TS.extra_ops.repeat(mask, self.hidden_dim, axis=2)
        # Вставляем, используя switch
        result = K.switch(mask, tensor, data)
        return result



    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'hidden_dim': self.hidden_dim,
                  'action_dim': self.action_dim,
                  'batch_size': self.batch_size,
                  'max_len': self.max_len,
                  'dropout_w': self.dropout_w,
                  'dropout_u': self.dropout_u,
                  'dropout_action': self.dropout_action,
                  'depth': self.depth}
        base_config = super(Encoder_Base, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
