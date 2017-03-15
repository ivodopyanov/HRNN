import keras.backend as K
from keras.layers import initializations, activations
from keras.engine import Layer

import numpy as np
import theano as T
import theano.tensor as TS
from theano.ifelse import ifelse
from theano.printing import Print


class Encoder(Layer):
    def __init__(self, input_dim, hidden_dim, action_dim, depth, batch_size, max_len,
                 dropout_w, dropout_u, dropout_action,
                 init='glorot_uniform', inner_init='orthogonal', **kwargs):
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
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.supports_masking = True
        self.gamma_init = initializations.get('one')
        self.beta_init = initializations.get('zero')
        self.epsilon = 1e-5
        self.mode = 0

        if self.dropout_w or self.dropout_u or self.dropout_action:
            self.uses_learning_phase = True

        super(Encoder, self).__init__(**kwargs)



    def build(self, input_shape):
        self.W_emb = self.init((self.input_dim, self.hidden_dim), name='{}_W_emb'.format(self.name))
        self.b_emb = K.zeros((self.hidden_dim), name='{}_b_emb'.format(self.name))

        self.W = self.init((self.hidden_dim, 3*self.hidden_dim), name='{}_W'.format(self.name))
        self.U = self.inner_init((self.hidden_dim, 3*self.hidden_dim), name='{}_U'.format(self.name))
        self.b = K.zeros((3*self.hidden_dim), name='{}_b'.format(self.name))


        self.W_action_1 = self.init((self.hidden_dim, self.action_dim), name='{}_W_action_1'.format(self.name))
        self.U_action_1 = self.inner_init((self.hidden_dim, self.action_dim), name='{}_U_action_1'.format(self.name))
        self.b_action_1 = K.zeros((self.action_dim,), name='{}_b_action_1'.format(self.name))

        self.W_action_2 = K.zeros((self.action_dim,2), name='{}_W_action_2'.format(self.name))
        self.b_action_2 = K.variable([1, -1], name='{}_b_action_2'.format(self.name))

        self.gammas = K.ones((2, 3*self.hidden_dim,), name="gammas")
        self.betas = K.zeros((2, 3*self.hidden_dim,), name="betas")
        self.trainable_weights = [self.W_emb, self.b_emb, self.W ,self.U , self.b, self.gammas, self.betas]
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], self.hidden_dim)


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

        initial_action = TS.zeros_like(x[:,:,0], dtype="int8")

        first_mask = K.zeros_like(data_mask[0])
        first_mask = K.expand_dims(first_mask, 0)
        eos_mask = K.concatenate([data_mask[1:], first_mask], axis=0)
        eos_mask = TS.cast(data_mask*(1-eos_mask), "int8")



        results, _ = T.scan(self.vertical_step,
                        outputs_info=[x, initial_action, data_mask],
                        non_sequences=[bucket_size, eos_mask, K.zeros((self.batch_size), dtype="int8")],
                        n_steps=self.depth-1)
        x = results[0][-1]
        initial_action = results[1][-1]
        data_mask = results[2][-1]


        results, _ = T.scan(self.vertical_step,
                            outputs_info=[x, initial_action, data_mask],
                            non_sequences=[bucket_size, eos_mask, K.ones((self.batch_size), dtype="int8")],
                            n_steps=1)

        outputs = results[0]
        outputs = outputs[-1,-1,:,:]
        return outputs

    # Vertical pass along hierarchy dimension
    def vertical_step(self, *args):
        x = args[0]
        action_prev = args[1]
        data_mask = args[2]
        bucket_size=args[3]
        eos_mask = args[4]
        last_layer_mask3 = args[5]


        initial_h = K.zeros((self.batch_size, self.hidden_dim))
        initial_action = K.zeros((self.batch_size), dtype="int8")
        initial_data_mask = K.zeros((self.batch_size), dtype="int8")
        initial_both_output = K.zeros((self.batch_size), dtype="int8")

        shifted_data_mask = K.concatenate([K.ones((1, self.batch_size)), data_mask[:-1]], axis=0)
        shifted_eos_mask = K.concatenate([K.zeros((1, self.batch_size)), eos_mask[:-1]], axis=0)


        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, action_prev, shifted_data_mask, data_mask, shifted_eos_mask, eos_mask],
                            outputs_info=[initial_h, initial_action, initial_data_mask, initial_both_output],
                            non_sequences=[last_layer_mask3],
                            n_steps=bucket_size)
        h = results[0]
        action = results[1]
        new_data_mask = results[2]
        both_output = results[3]

        #Shift computed action for 1 step left because at the step i we compute action for i-1
        last_action = K.zeros_like(action[0])
        last_action = K.expand_dims(last_action, 0)
        shifted_action = K.concatenate([action[1:], last_action], axis=0)
        shifted_action = TS.unbroadcast(shifted_action, 0, 1)
        # Uncomment to monitor action values during testing

        #shifted_action = Print("shifted_action")(shifted_action)
        #has_value = Print("has_value")(has_value)
        return [h, shifted_action, new_data_mask], T.scan_module.until(TS.eq(TS.sum(both_output), 0))

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
        last_layer_mask3 = args[10]


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
        policy = K.minimum(K.exp(K.dot(policy*B_action, self.W_action_2)+self.b_action_2),1000)

        action = K.switch(TS.le(policy[:,0], policy[:, 1]), 1, 0)
        action = K.switch(action_prev, 1, action)
        action = K.switch(last_layer_mask3, 1, action)
        action = K.switch(eos_mask_tm1, 0, action)
        action = TS.cast(action, "int8")

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
        both_output = TS.cast(both, "int8")

        data_mask = both + x_only + h_tm1_only
        data_mask = TS.cast(data_mask, "int8")

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
        data_mask_prev = data_mask_prev.dimshuffle([0,'x'])
        data_mask_prev = TS.extra_ops.repeat(data_mask_prev, self.hidden_dim, axis=1)
        h = K.switch(data_mask_prev, h, h_tm1)

        result = [h, action, data_mask, both_output]
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
                  'depth': self.depth,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__}
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
