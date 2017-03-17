import keras.backend as K
from keras.layers import initializations, activations, regularizers
from keras.engine import Layer

import theano.tensor as TS

class RL_Layer(Layer):
    def __init__(self,hidden_dim, action_dim,
                 dropout_w, dropout_u, dropout_action, l2,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 **kwargs):
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.dropout_w = dropout_w
        self.dropout_u = dropout_u
        self.dropout_action = dropout_action
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.l2 = l2
        if self.dropout_w or self.dropout_u or self.dropout_action:
            self.uses_learning_phase = True
        super(RL_Layer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.W_action_1 = self.init((self.hidden_dim, self.action_dim), name='{}_W_action_1'.format(self.name))
        self.U_action_1 = self.inner_init((self.hidden_dim, self.action_dim), name='{}_U_action_1'.format(self.name))
        self.b_action_1 = K.zeros((self.action_dim,), name='{}_b_action_1'.format(self.name))

        self.W_action_2 = K.zeros((self.action_dim,2), name='{}_W_action_2'.format(self.name))
        self.b_action_2 = K.variable([1, -1], name='{}_b_action_2'.format(self.name))

        self.trainable_weights = [self.W_action_1, self.U_action_1, self.b_action_1, self.W_action_2, self.b_action_2]
        if self.l2 is not None:
            for weight in self.trainable_weights:
                reg = regularizers.l2(self.l2)
                reg.set_param(weight)
                self.regularizers.append(reg)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
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

        policy = activations.relu(K.dot(x*B_W, self.W_action_1) + K.dot(h_tm1*B_U, self.U_action_1) + self.b_action_1)
        policy = K.minimum(K.exp(K.dot(policy*B_action, self.W_action_2)+self.b_action_2),1000)
        return policy

    def get_config(self):
        config = {'hidden_dim': self.hidden_dim,
                  'action_dim': self.action_dim,
                  'dropout_action': self.dropout_action,
                  'dropout_w': self.dropout_w,
                  'dropout_u': self.dropout_u,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__}
        base_config = super(RL_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))