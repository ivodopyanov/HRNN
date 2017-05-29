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

        x = K.dot(x, self.W_emb) + self.b_emb
        x = x.dimshuffle([1,0,2])
        x = x[:bucket_size]

        initial_h = K.zeros((self.batch_size, self.hidden_dim))
        initial_depth = K.zeros((1,), dtype="int8")

        results, _ = T.scan(self.vertical_step,
                        outputs_info=[x, data_mask, initial_depth],
                        non_sequences=[bucket_size],
                        n_steps=self.depth-1)
        x = results[0][-1]
        data_mask = results[1][-1]
        depth = results[2][-1]


        results, _ = T.scan(self.final_step,
                            sequences=[x, data_mask],
                            outputs_info=[initial_h])

        outputs = results[-1]
        return [outputs, depth]


    def vertical_step(self, x, x_mask, prev_depth, bucket_size):
        initial_h = x[0]
        initial_total_h = K.zeros_like(x)
        initial_total_h = initial_total_h.dimshuffle([1,0,2])
        initial_total_h_mask = K.zeros_like(x_mask)
        initial_total_h_mask = initial_total_h_mask.dimshuffle([1,0])

        initial_x_mask_tm1 = x_mask[0]

        results, _ = T.scan(self.horizontal_step,
                            sequences=[x[1:], x_mask[1:]],
                            outputs_info=[initial_h, initial_total_h, initial_total_h_mask, initial_x_mask_tm1],
                            non_sequences=[bucket_size])
        total_h = results[1]
        total_h_mask = results[2]

        depth = TS.cast(prev_depth+1, dtype="int8")

        return [total_h[-1].dimshuffle([1,0,2]), total_h_mask[-1].dimshuffle([1,0]), depth], T.scan_module.until(TS.eq(TS.sum(total_h_mask), TS.sum(x_mask)))

    def horizontal_step(self, x, x_mask, h_tm1, total_h_tm1, total_h_mask_tm1, x_mask_tm1, bucket_size):
        total_h_mask_next = self.get_next_value_mask(total_h_mask_tm1)

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

        policy = activations.relu(K.dot(x*B_W, self.W_action_1) + K.dot(h_tm1*B_U, self.U_action_1) + self.b_action_1)
        policy = K.exp(K.softmax(K.dot(policy*B_action, self.W_action_3)+self.b_action_3))

        continue_accumulation = K.switch(TS.le(policy[:,0], policy[:, 1]), 1, 0)
        continue_accumulation = K.switch(x_mask_tm1*(1-x_mask), 0, continue_accumulation)
        continue_accumulation = TS.cast(continue_accumulation, "bool")


        total_h_after_reduce = self.insert_tensor_at_mask(total_h_tm1, total_h_mask_next, h_tm1, bucket_size)
        continue_accumulation_for_total_h = continue_accumulation.dimshuffle([0,'x'])
        continue_accumulation_for_total_h = TS.extra_ops.repeat(continue_accumulation_for_total_h, bucket_size, axis=1)
        total_h_mask = K.switch(continue_accumulation_for_total_h, total_h_mask_tm1, total_h_mask_tm1 + total_h_mask_next)
        continue_accumulation_for_total_h = continue_accumulation_for_total_h.dimshuffle([0,1,'x'])
        continue_accumulation_for_total_h = TS.extra_ops.repeat(continue_accumulation_for_total_h, self.hidden_dim, axis=2)
        total_h = K.switch(continue_accumulation_for_total_h, total_h_tm1, total_h_after_reduce)

        h_ = K.relu(K.dot(x*B_W, self.W) + K.dot(h_tm1*B_U, self.U) + self.b)
        continue_accumulation_for_h = continue_accumulation.dimshuffle([0,'x'])
        continue_accumulation_for_h = TS.extra_ops.repeat(continue_accumulation_for_h, self.hidden_dim, axis=1)
        h = K.switch(continue_accumulation_for_h, h_, x)

        copy_old_value_mask = (1-x_mask)*(1-x_mask_tm1)
        copy_old_value_mask = copy_old_value_mask.dimshuffle([0,'x'])
        copy_old_value_mask = TS.extra_ops.repeat(copy_old_value_mask, bucket_size, axis=1)
        total_h_mask = K.switch(copy_old_value_mask, total_h_mask_tm1, total_h_mask)
        copy_old_value_mask = copy_old_value_mask.dimshuffle([0,1,'x'])
        copy_old_value_mask = TS.extra_ops.repeat(copy_old_value_mask, self.hidden_dim, axis=2)
        total_h = K.switch(copy_old_value_mask, total_h_tm1, total_h)

        return h, total_h, total_h_mask, x_mask


    def get_config(self):
        base_config = super(Encoder_Processor, self).get_config()
        return base_config.items()
