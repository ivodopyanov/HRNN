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
from Encoder_Base import Encoder_Base


class Encoder_Processor(Encoder_Base):
    def __init__(self, **kwargs):
        super(Encoder_Processor, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return [None, None]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.hidden_dim),
                (1,)]


    def call(self, input, mask=None):
        x = input[0]

        # Keras doesn't allow 1D model inputs - so that tensor have shape (1,1) instead of scalar or (1,)
        bucket_size = input[1][0][0]


        data_mask = mask[0]
        if data_mask.ndim == x.ndim-1:
            data_mask = K.expand_dims(data_mask)
        assert data_mask.ndim == x.ndim
        data_mask = data_mask.dimshuffle([1,0])
        data_mask = data_mask[:bucket_size]

        #data_mask = Print("data_mask")(data_mask)

        x = K.dot(x, self.W_emb) + self.b_emb
        x = x.dimshuffle([1,0,2])
        x = x[:bucket_size]

        initial_h = K.zeros((self.batch_size, self.hidden_dim))
        initial_depth = K.zeros((1,), dtype="int8")

        #x = Print("x")(x)
        #data_mask = Print("data_mask")(data_mask)

        results, _ = T.scan(self.vertical_step,
                        outputs_info=[x, data_mask, data_mask, initial_depth],
                        non_sequences=[bucket_size],
                        n_steps=self.depth-1)
        x = results[0][-1]
        data_mask = results[1][-1]
        has_value = results[2][-1]
        depth = results[3][-1]

        #data_mask = Print("final_data_mask")(data_mask)
        #has_value = Print("final has_value")(has_value)
        initial_final_has_value = K.zeros((self.batch_size), dtype="bool")
        results, _ = T.scan(self.final_step,
                            sequences=[x, data_mask, has_value],
                            outputs_info=[initial_h, initial_final_has_value])

        outputs = results[0][-1]
        return [outputs, depth]


    def vertical_step(self, x, x_mask, prev_has_value, prev_depth, bucket_size):

        #x_mask = Print("x_mask")(x_mask)
        #prev_has_value = Print("prev_has_Value")(prev_has_value)

        initial_h = K.zeros((self.batch_size, self.hidden_dim), name="initial_h")
        initial_new_mask = K.ones((self.batch_size), dtype="bool", name="initial_new_mask")
        initial_has_value = K.zeros((self.batch_size), dtype="bool")
        initial_both = K.zeros((self.batch_size), dtype="bool")

        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, x_mask, prev_has_value],
                            outputs_info=[initial_h, initial_new_mask, initial_has_value, initial_both])
        new_h = results[0]
        new_mask = results[1]
        has_value = results[2]
        both = results[3]

        new_mask = TS.concatenate([new_mask[1:], K.ones((1, self.batch_size), dtype="bool")], axis=0)

        depth = TS.cast(prev_depth+1, dtype="int8")


        #new_mask = Print("new_mask")(new_mask)
        #has_value = Print("has_value")(has_value)
        #new_h = Print("h")(new_h)

        return [new_h, new_mask, has_value, depth], T.scan_module.until(TS.eq(TS.sum(both), 0))

    def horizontal_step(self, x, prev_mask, prev_has_value, h_tm1, new_mask_tm1, has_value_tm1, both_tm1):

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
        if 0 < self.dropout_action < 1:
            ones = K.ones((self.action_dim))
            B_action = K.in_train_phase(K.dropout(ones, self.dropout_action), ones)
        else:
            B_action = K.cast_to_floatx(1.)
        if 0 < self.dropout_w < 1:
            ones = K.ones((self.hidden_dim))
            B_W1 = K.in_train_phase(K.dropout(ones, self.dropout_w), ones)
        else:
            B_W1 = K.cast_to_floatx(1.)

        policy = activations.relu(K.dot(x*B_W, self.W_action_1) + K.dot(h_tm1*B_U, self.U_action_1) + self.b_action_1)
        policy = K.exp(K.minimum(K.dot(policy*B_action, self.W_action_3)+self.b_action_3,5))
        #policy = K.exp(K.minimum(K.dot(x*B_W, self.W_action_1) + K.dot(h_tm1*B_U, self.U_action_1) + self.b_action_1, 5))

        # 1 = reduce, 0 = continue acc
        new_mask = K.switch(TS.le(policy[:,0], policy[:, 1]), 1, 0)
        new_mask = prev_mask*new_mask
        new_mask = TS.cast(new_mask, "bool")

        #prev_mask = Print("prev_mask")(prev_mask)
        #prev_has_value = Print("prev_has_value")(prev_has_value)
        #new_mask = Print("new_mask")(new_mask)
        #has_value_tm1 = Print("has_value_tm1")(has_value_tm1)

        both = prev_mask*prev_has_value*(1-new_mask)*has_value_tm1
        x_only = prev_mask*prev_has_value*(new_mask + (1-new_mask)*(1-has_value_tm1))
        h_only = (1-prev_mask + prev_mask*(1-prev_has_value))*(1-new_mask)*has_value_tm1

        #both = Print("both")(both)
        #x_only = Print("x_only")(x_only)
        #h_only = Print("h_only")(h_only)

        has_value = both + x_only + h_only
        has_value = TS.cast(has_value, "bool")


        both_for_h = both.dimshuffle([0,'x'])
        both_for_h = TS.extra_ops.repeat(both_for_h, self.hidden_dim, axis=1)
        x_only_for_h = x_only.dimshuffle([0,'x'])
        x_only_for_h = TS.extra_ops.repeat(x_only_for_h, self.hidden_dim, axis=1)
        h_only_for_h = h_only.dimshuffle([0,'x'])
        h_only_for_h = TS.extra_ops.repeat(h_only_for_h, self.hidden_dim, axis=1)

        h_ = activations.relu(K.dot(x*B_W, self.W) + K.dot(h_tm1*B_U, self.U) + self.b)
        h_ = activations.relu(K.dot(h_*B_W1, self.W1) + self.b1)
        h = both_for_h*h_ + x_only_for_h*x + h_only_for_h*h_tm1

        return h, new_mask, has_value, TS.cast(both, "bool")


    def get_config(self):
        base_config = super(Encoder_Processor, self).get_config()
        return base_config.items()
