import keras.backend as K
from keras.layers import initializations, activations
from keras.engine import Layer

import theano.tensor as TS

class RL_Layer(Layer):
    def __init__(self,hidden_dim, action_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 **kwargs):
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(RL_Layer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.W_action_1 = self.init((self.hidden_dim, self.action_dim), name='{}_W_action_1'.format(self.name))
        self.U_action_1 = self.inner_init((self.hidden_dim, self.action_dim), name='{}_U_action_1'.format(self.name))
        self.b_action_1 = K.zeros((self.action_dim,), name='{}_b_action_1'.format(self.name))

        self.W_action_2 = self.init((self.action_dim,2), name='{}_W_action_2'.format(self.name))
        self.b_action_2 = K.zeros((2,), name='{}_b_action_2'.format(self.name))

        self.trainable_weights = [self.W_action_1, self.U_action_1, self.b_action_1, self.W_action_2, self.b_action_2]

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 2)

    def call(self, input, mask=None):
        x = input[0]
        h_tm1 = input[1]

        policy = activations.relu(K.dot(x, self.W_action_1) + K.dot(h_tm1, self.U_action_1) + self.b_action_1)
        policy = TS.exp(K.dot(policy, self.W_action_2)+self.b_action_2)
        return policy
