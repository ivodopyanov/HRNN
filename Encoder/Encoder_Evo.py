# -*- coding: utf-8 -*-
import keras.backend as K
from keras import activations
from keras.engine import Layer


import theano as T
import theano.tensor as TS
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import Print
from Encoder.Encoder_Base import Encoder_Base


class Encoder_Evo(Encoder_Base):
    def __init__(self, sigma, **kwargs):
        self.sigma = sigma
        super(Encoder_Evo, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return [None, None, None, None, None, None, None]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.hidden_dim),
                (self.hidden_dim, self.action_dim),
                (self.action_dim, self.action_dim),
                (self.action_dim,),
                (self.action_dim, 2),
                (2, ),
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




        srng = RandomStreams()
        W_action_1_rnd = srng.normal((self.hidden_dim, self.action_dim))
        U_action_1_rnd = srng.normal((self.action_dim, self.action_dim))
        b_action_1_rnd = srng.normal((self.action_dim, ))
        W_action_2_rnd = srng.normal((self.action_dim, 2))
        b_action_2_rnd = srng.normal((2, ))


        initial_h = K.zeros((self.batch_size, self.hidden_dim))
        initial_depth = K.zeros((1,), dtype="int8")

        results, _ = T.scan(self.vertical_step,
                        outputs_info=[x, data_mask, initial_depth],
                        non_sequences=[bucket_size, W_action_1_rnd, U_action_1_rnd, b_action_1_rnd, W_action_2_rnd, b_action_2_rnd],
                        n_steps=self.depth-1)
        x = results[0][-1]
        data_mask = results[1][-1]
        depth = results[2][-1]



        results, _ = T.scan(self.final_step,
                            sequences=[x, data_mask],
                            outputs_info=[initial_h])

        outputs = results[-1]
        return [outputs,
                W_action_1_rnd,
                U_action_1_rnd,
                b_action_1_rnd,
                W_action_2_rnd,
                b_action_2_rnd,
                depth]

    def vertical_step(self, x, x_mask, prev_depth, bucket_size, W_action_1_rnd, U_action_1_rnd, b_action_1_rnd, W_action_2_rnd, b_action_2_rnd):
        initial_h = x[0]
        initial_total_h = K.zeros_like(x)
        initial_total_h = initial_total_h.dimshuffle([1,0,2])
        initial_total_h_mask = K.zeros_like(x_mask)
        initial_total_h_mask = initial_total_h_mask.dimshuffle([1,0])
        initial_x_mask_tm1 = x_mask[0]


        results, _ = T.scan(self.horizontal_step,
                            sequences=[x[1:], x_mask[1:]],
                            outputs_info=[initial_h, initial_total_h, initial_total_h_mask, initial_x_mask_tm1],
                            non_sequences=[bucket_size, W_action_1_rnd, U_action_1_rnd, b_action_1_rnd, W_action_2_rnd, b_action_2_rnd])
        total_h = results[1][-1]
        total_h_mask = results[2][-1]

        depth = TS.cast(prev_depth+1, dtype="int8")

        return [total_h.dimshuffle([1,0,2]), total_h_mask.dimshuffle([1,0]), depth], T.scan_module.until(TS.eq(TS.sum(total_h_mask), TS.sum(x_mask)))



    def horizontal_step(self, x, x_mask, h_tm1, total_h_tm1, total_h_mask_tm1, x_mask_tm1, bucket_size, W_action_1_rnd, U_action_1_rnd, b_action_1_rnd, W_action_2_rnd, b_action_2_rnd):
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

        policy = activations.relu(K.dot(x*B_W, self.W_action_1+self.sigma*W_action_1_rnd) + K.dot(h_tm1*B_U, self.U_action_1+self.sigma*U_action_1_rnd) + self.b_action_1+self.sigma*b_action_1_rnd)
        #policy = activations.relu(K.dot(policy*B_action, self.W_action_2)+self.b_action_2)
        #policy = K.exp(K.minimum(K.dot(policy*B_action, self.W_action_3)+self.b_action_3,5))
        policy = K.exp(K.dot(policy*B_action, self.W_action_3+self.sigma*W_action_2_rnd)+self.b_action_3+self.sigma*b_action_2_rnd)

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

        h_ = self.gru_step(x, h_tm1, B_W, B_U)
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
        config = {}
        base_config = super(Encoder_Evo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
