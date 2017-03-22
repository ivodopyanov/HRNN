import keras.backend as K
from keras.layers import activations
from keras.engine import Layer

from keras.initializers import glorot_uniform, orthogonal, ones, zeros
import numpy as np
import theano as T
import theano.tensor as TS
from theano.ifelse import ifelse
from theano.printing import Print


class Decoder_Processor(Layer):
    def __init__(self, hidden_dim, output_dim, action_dim, depth, batch_size, max_len,
                 dropout_w, dropout_u, dropout_action, **kwargs):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.depth = depth
        self.batch_size = batch_size
        self.max_len = max_len
        self.dropout_w = dropout_w
        self.dropout_u = dropout_u
        self.dropout_action = dropout_action
        self.supports_masking = True
        self.epsilon = 1e-5

        if self.dropout_w or self.dropout_u or self.dropout_action:
            self.uses_learning_phase = True

        super(Decoder_Processor, self).__init__(**kwargs)



    def build(self, input_shape):

        self.W = self.add_weight((self.hidden_dim, 3*self.hidden_dim),
                                 initializer=glorot_uniform(),
                                 name='{}_W'.format(self.name))
        self.U = self.add_weight((self.hidden_dim, 3*self.hidden_dim),
                                 initializer=orthogonal(),
                                 name='{}_U'.format(self.name))
        self.b = self.add_weight((3*self.hidden_dim),
                                 initializer=zeros(),
                                 name='{}_b'.format(self.name))


        self.W_action_1 = self.add_weight((self.hidden_dim, self.action_dim),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='{}_W_action_1'.format(self.name))
        self.U_action_1 = self.add_weight((self.hidden_dim, self.action_dim),
                                          initializer=orthogonal(),
                                          trainable=False,
                                          name='{}_U_action_1'.format(self.name))
        self.b_action_1 = self.add_weight((self.action_dim,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='{}_b_action_1'.format(self.name))
        self.W_action_2 = self.add_weight((self.action_dim,2),
                                          initializer=glorot_uniform(),
                                          trainable=False,
                                          name='{}_W_action_2'.format(self.name))
        self.b_action_2 = self.add_weight((2,),
                                          initializer=zeros(),
                                          trainable=False,
                                          name='{}_b_action_2'.format(self.name))

        self.gammas = self.add_weight((2, 3*self.hidden_dim,),
                                      initializer=ones(),
                                      name="gammas")
        self.betas = self.add_weight((2, 3*self.hidden_dim,),
                                     initializer=zeros(),
                                     name="betas")
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.max_len, self.hidden_dim)


    def call(self, input, mask=None):
        x = input[0]

        initial_h = K.zeros((self.depth, self.hidden_dim))
        initial_action = K.zeros((self.depth,), dtype="int8")

        results, _ = T.scan(self.horizontal_step,
                        outputs_info=[initial_h, initial_action],
                        non_sequences=[x],
                        n_steps=self.max_len)

        outputs = results[0][-1]
        return outputs

    def horizontal_step(self, *args):
        h_tm1 = args[0]
        action_tm1 = args[1]
        x = args[2]

        h_for_action = K.concatenate([x, h_tm1], axis=1)
        initial_action = K.ones((self.batch_size), dtype="bool")


        results, _ = T.scan(self.vertical_step_action,
                            sequences=[{'initial': h_for_action, 'taps':[-1,0]}],
                            outputs_info=[initial_action],
                            go_backwards=True,
                            n_steps=self.depth)

        action = results[0]
        last_action = K.ones_like(action[-1])
        last_action = K.expand_dims(last_action)
        process_values = K.concatenate([action[1:], last_action], axis=0)

        results, _ = T.scan(self.vertical_step_h,
                            sequences=[action, process_values, h_tm1],
                            outputs_info=[x],
                            n_steps=self.depth)

        h = results[0][-1]
        action = results[1]
        return h, action

    # 1 = закончили генерить h по текущему контексту prev_h
    # 0 = продолжаем генерить
    def vertical_step_action(self, prev_h_tm1, h_tm1, action_tm1):
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

        policy = activations.relu(K.dot(prev_h_tm1*B_W, self.W_action_1) + K.dot(h_tm1*B_U, self.U_action_1) + self.b_action_1)
        policy = K.minimum(K.exp(K.dot(policy*B_action, self.W_action_2)+self.b_action_2),1000)
        action = K.switch(TS.le(policy[:,0], policy[:, 1]), 1, 0)
        action = K.switch(action_tm1, action, 0)
        action = TS.cast(action, "bool")
        return action


    def vertical_step_h(self, action, process_values, h_tm1, x):
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

        h_tm1 = K.switch(action, K.zeros_like(h_tm1), h_tm1)

        s1 = self.ln(K.dot(x*B_W, self.W) + self.b, self.gammas[0], self.betas[0])
        s2 = self.ln(K.dot(h_tm1*B_U, self.U[:,:2*self.hidden_dim]), self.gammas[1,:2*self.hidden_dim], self.betas[1,:2*self.hidden_dim])
        s = K.hard_sigmoid(s1[:,:2*self.hidden_dim] + s2)
        z = s[:,:self.hidden_dim]
        r = s[:,self.hidden_dim:2*self.hidden_dim]
        h_ = z*h_tm1 + (1-z)*K.tanh(s1[:,2*self.hidden_dim:] + self.ln(K.dot(r*h_tm1*B_U, self.U[:,2*self.hidden_dim:]), self.gammas[1,2*self.hidden_dim:], self.betas[1,2*self.hidden_dim:]))

        h = K.switch(process_values, h_, h_tm1)
        return h





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
        base_config = super(Decoder_Processor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
