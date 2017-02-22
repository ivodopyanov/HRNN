import keras.backend as K
from keras.layers import initializations, activations
from keras.engine import Layer

class FK_Layer(Layer):
    def __init__(self,input_dim, hidden_dim, FK_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 **kwargs):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.FK_dim = FK_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(FK_Layer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.W_FK_1 = self.init((self.input_dim+self.hidden_dim, self.FK_dim), name='{}_W_FK_1'.format(self.name))
        self.U_FK_1 = self.inner_init((self.input_dim+self.hidden_dim, self.FK_dim), name='{}_U_FK_1'.format(self.name))
        self.b_FK_1 = K.zeros((self.FK_dim,), name='{}_b_FK_1'.format(self.name))
        self.action_FK_right = K.zeros((1,self.FK_dim), name='{}_ACTION_FK_right'.format(self.name))
        self.action_FK_top = K.zeros((1,self.FK_dim), name='{}_ACTION_FK_top'.format(self.name))

        self.W_FK_2 = self.init((self.FK_dim, 1), name='{}_W_FK_2'.format(self.name))
        self.b_FK_2 = K.zeros((1,), name='{}_b_FK_2'.format(self.name))

        self.trainable_weights = [self.W_FK_1, self.U_FK_1, self.b_FK_1, self.action_FK_right, self.action_FK_top, self.W_FK_2, self.b_FK_2]

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)

    def call(self, input, mask=None):
        x = input[0]
        h_tm1 = input[1]
        action = input[2]

        sum1_fk = K.dot(x, self.W_FK_1)
        sum2_fk = K.dot(h_tm1, self.U_FK_1)



        sum_fk = sum1_fk + sum2_fk + self.b_FK_1 + K.dot(action,self.action_FK_right) + K.dot(1-action, self.action_FK_top)

        fk_candidate_1 = self.inner_activation(sum_fk)
        fk_candidate_2 = self.inner_activation(K.dot(fk_candidate_1,self.W_FK_2)+self.b_FK_2)
        return fk_candidate_2
