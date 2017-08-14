# -*- coding: utf-8 -*-
import keras.backend as K
from keras import activations
from keras.engine import Layer


import theano as T
import theano.tensor as TS
from theano.printing import Print
from Encoder4.Encoder_Base import Encoder_Base


class Encoder_Predictor(Encoder_Base):
    def __init__(self, **kwargs):
        self.uses_learning_phase = True
        super(Encoder_Predictor, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return [None, None, None, None, None, None, None, None]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.hidden_dim),
                (self.batch_size, self.depth, 1,self.hidden_dim),
                (self.batch_size, self.depth, 1,self.hidden_dim),
                (self.batch_size, self.depth, 1, 2),
                (self.batch_size, self.depth, 1,),
                (self.batch_size, self.depth, 1,),
                (1,),
                (self.batch_size, self.depth, 1,)]


    def call(self, input, mask=None):
        x = input[0]

        # Keras doesn't allow 1D model inputs - so that tensor have shape (1,1) instead of scalar or (1,)
        bucket_size = input[1][0][0]
        random_action_prob = input[2][0][0]


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

        #x = Print("x")(x)


        initial_h = K.zeros((self.batch_size, self.hidden_dim))
        initial_policy = K.zeros_like(x)
        initial_policy = K.sum(initial_policy, axis=2)
        initial_policy = initial_policy.dimshuffle([0,1,'x'])
        initial_policy = TS.extra_ops.repeat(initial_policy, 2, axis=2)

        initial_policy_input_x = K.zeros_like(x)
        initial_policy_input_h = K.zeros_like((x))
        initial_depth = K.zeros((1,), dtype="int8")
        initial_policy_depth = K.zeros_like(data_mask, dtype="int8")

        initial_thread_id, _ = T.scan(self.init_thread_id,
                                      sequences=[data_mask],
                                      outputs_info=[K.zeros((self.batch_size,), dtype="int64")])
        initial_maximum_id = TS.sum(data_mask, axis=0)

        results, _ = T.scan(self.vertical_step,
                        outputs_info=[x, data_mask, data_mask, initial_policy_input_x, initial_policy_input_h, initial_policy, initial_policy_depth, initial_depth, initial_thread_id, initial_maximum_id],
                        non_sequences=[random_action_prob],
                        n_steps=self.depth-1)

        chosen_action = results[1]
        has_value = results[2]
        policy_input_x = results[3]
        policy_input_h = results[4]
        policy = results[5]
        policy_depth = results[6]
        depth = results[7][-1]
        thread_id = results[8]

        outputs = results[0][-1][-1]

        return [outputs,
                policy_input_x.dimshuffle([2,0,1,3]),
                policy_input_h.dimshuffle([2,0,1,3]),
                policy.dimshuffle([2,0,1,3]),
                chosen_action.dimshuffle([2,0,1]),
                policy_depth.dimshuffle([2,0,1]),
                depth,
                thread_id.dimshuffle([2,0,1])]

    def init_thread_id(self, mask, prev_id):
        return TS.switch(mask, prev_id+1, 0)

    def vertical_step(self, x, x_mask, prev_has_value, policy_input_x_tm1, policy_input_h_tm1, policy_tm1, prev_policy_depth, prev_depth, prev_thread_id, prev_maximum_id, random_action_prob):

        initial_h = K.zeros((self.batch_size, self.hidden_dim), name="initial_h")
        initial_new_mask = K.ones((self.batch_size), dtype="bool", name="initial_new_mask")
        initial_policy = K.zeros((self.batch_size, 2), name="initial_policy")
        initial_policy_input_x = K.zeros((self.batch_size, self.hidden_dim), name="initial_policy_input_x")
        initial_policy_input_h = K.zeros((self.batch_size, self.hidden_dim), name="initial_policy_input_h")
        initial_has_value = K.zeros((self.batch_size), dtype="bool")
        initial_policy_depth = K.zeros((self.batch_size), dtype="int8")
        initial_thread_id = K.zeros((self.batch_size), dtype="int64")
        initial_had_both_on_current_lebel = K.zeros((self.batch_size), dtype="bool")
        last_value_mask = TS.concatenate([TS.zeros_like(x_mask, dtype="int8")[:-1], TS.ones_like(x_mask, dtype="int8")[0:1]], axis=0)

        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, x_mask, prev_has_value, last_value_mask, prev_policy_depth, prev_thread_id],
                            outputs_info=[initial_h, initial_new_mask, initial_has_value, initial_had_both_on_current_lebel, initial_policy_input_x, initial_policy_input_h, initial_policy, initial_policy_depth, initial_thread_id, prev_maximum_id],
                            non_sequences=[random_action_prob])
        new_h = results[0]
        new_mask = results[1]
        has_value = results[2]
        policy_input_x = results[4]
        policy_input_h = results[5]
        policy = results[6]
        policy_depth = results[7]
        thread_id = results[8]
        maximum_id = results[9][-1]




        new_mask = TS.concatenate([new_mask[1:], K.ones((1, self.batch_size), dtype="bool")], axis=0)
        policy_input_x = TS.concatenate([policy_input_x[1:], K.zeros((1, self.batch_size, self.hidden_dim))], axis=0)
        policy_input_h = TS.concatenate([policy_input_h[1:], K.zeros((1, self.batch_size, self.hidden_dim))], axis=0)
        policy = TS.concatenate([policy[1:], K.zeros((1, self.batch_size, 2))], axis=0)
        policy_depth = TS.concatenate([policy_depth[1:], K.zeros((1, self.batch_size), dtype="int8")], axis=0)

        depth = TS.cast(prev_depth+1, dtype="int8")

        return [new_h, new_mask, has_value, policy_input_x, policy_input_h, policy, policy_depth, depth, thread_id, maximum_id], T.scan_module.until(TS.eq(TS.sum(new_mask*has_value), self.batch_size))

    def horizontal_step(self, x, prev_mask, prev_has_value, last_value_mask, prev_policy_depth, prev_thread_id, h_tm1, mask_tm1, has_value_tm1, had_both_on_current_level_tm1, policy_input_x_tm1, policy_input_h_tm1, policy_tm1, policy_depth_tm1, thread_id_tm1, maximum_id_tm1, random_action_prob):

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
        policy = K.exp(K.minimum(K.dot(policy*B_action, self.W_action_3)+self.b_action_3, 5))


        # 1 = reduce, 0 = continue acc
        new_mask = K.switch(TS.le(policy[:,0], policy[:, 1]), 1, 0)
        use_random_action = K.random_binomial((self.batch_size,), random_action_prob)
        use_random_action = K.in_train_phase(use_random_action, K.zeros((self.batch_size)))
        random_action = K.random_uniform((self.batch_size,)) >= 0.5
        new_mask = K.switch(use_random_action, random_action, new_mask)


        new_mask = prev_mask*prev_has_value*new_mask
        new_mask = K.switch(last_value_mask*(1-had_both_on_current_level_tm1), 0, new_mask)
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
        #x = Print("x")(x)
        #h_tm1 = Print("h_tm1")(h_tm1)

        has_value = both + x_only + h_only
        has_value = TS.cast(has_value, "bool")

        #has_value = Print("has_value")(has_value)


        both_for_h = both.dimshuffle([0,'x'])
        both_for_h = TS.extra_ops.repeat(both_for_h, self.hidden_dim, axis=1)
        x_only_for_h = x_only.dimshuffle([0,'x'])
        x_only_for_h = TS.extra_ops.repeat(x_only_for_h, self.hidden_dim, axis=1)
        h_only_for_h = h_only.dimshuffle([0,'x'])
        h_only_for_h = TS.extra_ops.repeat(h_only_for_h, self.hidden_dim, axis=1)

        h_ = self.gru_step(x, h_tm1,B_W,B_U,B_W1,self.W, self.U, self.b, self.inner_dim)

        h = both_for_h*h_ + x_only_for_h*x + h_only_for_h * h_tm1

        both = TS.cast(both, "bool")

        policy_depth = x_only*prev_policy_depth + h_only*policy_depth_tm1 + both*(K.maximum(prev_policy_depth, policy_depth_tm1)+1)
        maximum_id = K.switch(both, maximum_id_tm1+1, maximum_id_tm1)
        new_thread_id = both*maximum_id + x_only*prev_thread_id + h_only*thread_id_tm1

        had_both_on_current_level = TS.or_(both, had_both_on_current_level_tm1)

        return h, new_mask, has_value, had_both_on_current_level, x, h_tm1, policy, policy_depth, new_thread_id, maximum_id



    def get_config(self):
        config = {}
        base_config = super(Encoder_Predictor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
