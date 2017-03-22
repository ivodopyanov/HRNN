# -*- coding: utf-8 -*-
import keras.backend as K
from keras import activations
from keras.engine import Layer

from keras.initializers import glorot_uniform, orthogonal, zeros
from keras.regularizers import l2
import theano as T
import theano.tensor as TS
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams


class Encoder_Predictor(Layer):
    def __init__(self, input_dim, hidden_dim, action_dim, depth, batch_size, max_len, random_action_prob,
                 dropout_w, dropout_u, dropout_action, **kwargs):
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
        self.depth = depth
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_len = max_len
        self.random_action_prob = random_action_prob
        self.dropout_w = dropout_w
        self.dropout_u = dropout_u
        self.dropout_action = dropout_action
        self.supports_masking = True
        self.epsilon = 1e-5
        if self.dropout_w or self.dropout_u or self.dropout_action:
            self.uses_learning_phase = True
        super(Encoder_Predictor, self).__init__(**kwargs)



    def build(self, input_shape):
        self.W_emb = self.add_weight((self.input_dim, self.hidden_dim),
                                     initializer=glorot_uniform(),
                                     trainable=False,
                                     name='W_emb_{}'.format(self.name))
        self.b_emb = self.add_weight((self.hidden_dim),
                                     initializer=zeros(),
                                     trainable=False,
                                     name='b_emb_{}'.format(self.name))

        self.W = self.add_weight((self.hidden_dim, 3*self.hidden_dim),
                                 initializer=glorot_uniform(),
                                 trainable=False,
                                 name='W_{}'.format(self.name))
        self.U = self.add_weight((self.hidden_dim, 3*self.hidden_dim),
                                 initializer=orthogonal(),
                                 trainable=False,
                                 name='U_{}'.format(self.name))
        self.b = self.add_weight((3*self.hidden_dim),
                                 initializer=zeros(),
                                 trainable=False,
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

        self.W_action_2 = self.add_weight((self.action_dim, 2),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='W_action_2_{}'.format(self.name))
        self.b_action_2 = self.add_weight((2,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='b_action_2_{}'.format(self.name))

        self.gammas = self.add_weight((2, 3*self.hidden_dim,),
                                      initializer=zeros(),
                                      trainable=False,
                                      name='gammas_{}'.format(self.name))
        self.betas = self.add_weight((2, 3*self.hidden_dim,),
                                     initializer=zeros(),
                                     trainable=False,
                                     name='betas_{}'.format(self.name))

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return [None, None, None, None, None, None, None, None]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.hidden_dim),
                (self.batch_size, self.depth, 1, 1),
                (self.batch_size, self.depth, 1, 1),
                (self.batch_size, self.depth, 1,self.hidden_dim),
                (self.batch_size, self.depth, 1,self.hidden_dim),
                (self.batch_size, self.depth, 1,2),
                (1,1),
                (self.batch_size, self.depth, 1, 1)]


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

        initial_action = TS.zeros_like(x[:,:,0], dtype="bool")
        initial_action1 = TS.zeros_like(initial_action)
        initial_action_calculated = TS.zeros_like(x[:,:,0], dtype="bool")
        first_mask = K.zeros_like(data_mask[0])
        first_mask = K.expand_dims(first_mask, 0)
        eos_mask = K.concatenate([data_mask[1:], first_mask], axis=0)
        eos_mask = TS.cast(data_mask*(1-eos_mask), "bool")
        initial_policy = K.zeros_like(x[:,:,0])
        initial_policy = initial_policy.dimshuffle([0,1,'x'])
        initial_policy = TS.extra_ops.repeat(initial_policy, 2, axis=2)
        initial_policy_input_x = K.zeros_like(x)
        initial_policy_input_h = K.zeros_like(x)
        initial_depth = K.zeros((1,), dtype="int8")

        if self.depth > 1:
            results, _ = T.scan(self.vertical_step,
                                outputs_info=[x, initial_action, data_mask, initial_action_calculated, initial_policy_input_x, initial_policy_input_h, initial_policy, initial_depth, initial_action1],
                                non_sequences=[bucket_size, eos_mask, K.zeros((self.batch_size), dtype="bool")],
                                n_steps=self.depth-1)
            x = results[0][-1]
            initial_action = results[1][-1]
            data_mask = results[2][-1]
            initial_action_calculated = results[3][-1]
            initial_policy_input_x = results[4][-1]
            initial_policy_input_h = results[5][-1]
            initial_policy = results[6][-1]
            initial_depth = results[7][-1]
            initial_action1 = results[8][-1]

        last_layer_results, _ = T.scan(self.vertical_step,
                            outputs_info=[x, initial_action, data_mask, initial_action_calculated, initial_policy_input_x, initial_policy_input_h, initial_policy, initial_depth, initial_action1],
                            non_sequences=[bucket_size, eos_mask, K.zeros((self.batch_size), dtype="bool")],
                            n_steps=1)

        output = last_layer_results[0][-1, -1, :, :]
        if self.depth > 1:
            action = K.concatenate([results[1], last_layer_results[1]], axis=0).dimshuffle([2,0,1])
            action_calculated = K.concatenate([results[3], last_layer_results[3]], axis=0).dimshuffle([2,0,1])
            policy_input_x = K.concatenate([results[4], last_layer_results[4]], axis=0).dimshuffle([2,0,1,3])
            policy_input_h = K.concatenate([results[5], last_layer_results[5]], axis=0).dimshuffle([2,0,1,3])
            policy = K.concatenate([results[6], last_layer_results[6]], axis=0).dimshuffle([2,0,1,3])
            action1 = K.concatenate([results[8], last_layer_results[8]], axis=0).dimshuffle([2,0,1])
        else:
            action = last_layer_results[1].dimshuffle([2,0,1])
            action_calculated = last_layer_results[3].dimshuffle([2,0,1])
            policy_input_x = last_layer_results[4].dimshuffle([2,0,1,3])
            policy_input_h = last_layer_results[5].dimshuffle([2,0,1,3])
            policy = last_layer_results[6].dimshuffle([2,0,1,3])
            action1 = last_layer_results[8].dimshuffle([2,0,1])
        depth = last_layer_results[7][-1]

        return [output, action, action_calculated, policy_input_x, policy_input_h, policy, depth, action1]

    # Vertical pass along hierarchy dimension
    def vertical_step(self, *args):
        x = args[0]
        action_prev = args[1]
        data_mask = args[2]
        action_calculated_prev = args[3]
        policy_input_x_prev = args[4]
        policy_input_h_prev = args[5]
        policy_prev = args[6]
        depth_prev = args[7]
        action1_prev = args[8]
        bucket_size=args[9]
        eos_mask = args[10]
        last_layer_mask3 = args[11]

        initial_h = K.zeros((self.batch_size, self.hidden_dim))
        initial_action = K.zeros((self.batch_size), dtype="bool")
        initial_action1 = K.zeros((self.batch_size), dtype="bool")
        initial_data_mask = K.zeros((self.batch_size), dtype="bool")
        initial_both_output = K.zeros((self.batch_size), dtype="bool")
        initial_action_calculated = K.zeros((self.batch_size), dtype="bool")
        initial_policy = K.zeros((self.batch_size, 2))

        shifted_data_mask = K.concatenate([K.ones((1, self.batch_size)), data_mask[:-1]], axis=0)
        shifted_eos_mask = K.concatenate([K.zeros((1, self.batch_size)), eos_mask[:-1]], axis=0)



        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, action_prev, shifted_data_mask, data_mask, shifted_eos_mask, eos_mask],
                            outputs_info=[initial_h, initial_action, initial_data_mask, initial_both_output, initial_action_calculated, initial_policy, initial_action1],
                            non_sequences=[last_layer_mask3],
                            n_steps=bucket_size)
        h = results[0]
        action = results[1]
        new_data_mask = results[2]
        both_output = results[3]
        action_calculated = results[4]
        policy = results[5]
        action1 = results[6]

        #Shift computed FK for 1 step left because at the step i we compute FK for i-1
        last_action = K.zeros_like(action[0])
        last_action = K.expand_dims(last_action, 0)
        shifted_action = K.concatenate([action[1:], last_action], axis=0)
        shifted_action = TS.unbroadcast(shifted_action, 0, 1)
        # Uncomment to monitor FK values during testing
        #shifted_action = Print("shifted_action")(shifted_action)
        #new_data_mask = Print("new_data_mask")(new_data_mask)
        policy_input_x = x
        policy_input_h = K.concatenate([K.zeros((1, self.batch_size, self.hidden_dim)), h[1:]], axis=0)

        new_depth = TS.cast(depth_prev+1, dtype="int8")

        return [h, shifted_action, new_data_mask, action_calculated, policy_input_x, policy_input_h, policy, new_depth, action1], T.scan_module.until(TS.eq(TS.sum(both_output), 0))

    # Horizontal pass along time dimension
    def horizontal_step(self, *args):
        x = args[0]
        action_prev = args[1]
        data_mask_prev_tm1 = args[2]
        data_mask_prev = args[3]
        eos_mask_tm1 = args[4]
        eos_mask = args[5]
        h_tm1 = args[6]
        action_tm1 = args[7]
        data_mask_tm1 = args[8]
        both_output_tm1 = args[9]
        action_calculated_tm1 = args[10]
        policy_tm1 = args[11]
        action1_tm1 = args[12]
        last_layer_mask3 = args[13]

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
        policy = K.exp(K.minimum(K.dot(policy*B_action, self.W_action_2)+self.b_action_2,5))

        action = K.switch(TS.le(policy[:,0], policy[:, 1]), 1, 0)
        action1 = TS.cast(action, "bool")

        srng = RandomStreams()
        select_random_action = TS.le(srng.uniform((self.batch_size,)), self.random_action_prob)
        random_action = TS.ge(srng.uniform((self.batch_size,)), 0.5)
        action = (1-select_random_action)*action + select_random_action*random_action

        action = K.switch(action_prev, 1, action)
        action = K.switch(last_layer_mask3, 1, action)
        action = K.switch(eos_mask_tm1, 0, action)
        action = TS.cast(action, "bool")

        action_calculated = data_mask_prev_tm1*(1-last_layer_mask3)*(1-eos_mask_tm1)*(1-action_prev)
        action_calculated = TS.cast(action_calculated, "bool")

        # Actual new hidden state if node got info from left and from below
        s1 = self.ln(K.dot(x*B_W, self.W) + self.b, self.gammas[0], self.betas[0])
        s2 = self.ln(K.dot(h_tm1*B_U, self.U[:,:2*self.hidden_dim]), self.gammas[1,:2*self.hidden_dim], self.betas[1,:2*self.hidden_dim])
        s = K.hard_sigmoid(s1[:,:2*self.hidden_dim] + s2)
        z = s[:,:self.hidden_dim]
        r = s[:,self.hidden_dim:2*self.hidden_dim]
        h_ = z*h_tm1 + (1-z)*K.tanh(s1[:,2*self.hidden_dim:] + self.ln(K.dot(r*h_tm1*B_U, self.U[:,2*self.hidden_dim:]), self.gammas[1,2*self.hidden_dim:], self.betas[1,2*self.hidden_dim:]))

        zeros = K.zeros((self.batch_size, self.hidden_dim))
        both = (1-action_prev)*data_mask_prev*action*data_mask_tm1
        h_tm1_only = data_mask_tm1*action*(action_prev + (1-action_prev)*(1-data_mask_prev))
        x_only = data_mask_prev*(1-action_prev)*((1-action) + action*(1-data_mask_tm1))
        both_output = TS.cast(both, "bool")

        data_mask = both + x_only + h_tm1_only
        data_mask = TS.cast(data_mask, "bool")

        both = both.dimshuffle([0,'x'])
        both = TS.extra_ops.repeat(both, self.hidden_dim, axis=1)
        h_tm1_only = h_tm1_only.dimshuffle([0,'x'])
        h_tm1_only = TS.extra_ops.repeat(h_tm1_only, self.hidden_dim, axis=1)
        x_only = x_only.dimshuffle([0,'x'])
        x_only = TS.extra_ops.repeat(x_only, self.hidden_dim, axis=1)

        h = K.switch(both, h_, zeros)
        h = K.switch(h_tm1_only, h_tm1, h)
        h = K.switch(x_only, x, h)

        # Apply mask

        action = K.switch(data_mask_prev_tm1, action, action_tm1)
        data_mask = data_mask_prev*data_mask
        data_mask_prev = data_mask_prev.dimshuffle([0,'x'])
        data_mask_prev = TS.extra_ops.repeat(data_mask_prev, self.hidden_dim, axis=1)
        h = K.switch(data_mask_prev, h, h_tm1)

        result = [h, action, data_mask, both_output, action_calculated, policy, action1]
        return result


    # Linear Normalization
    def ln(self, x, gammas, betas):
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
        x_normed = (x - m) / (std + self.epsilon)
        x_normed = gammas * x_normed + betas
        return x_normed



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
        base_config = super(Encoder_Predictor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
