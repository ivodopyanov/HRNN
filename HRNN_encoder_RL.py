import keras.backend as K
from keras.layers import initializations, activations
from keras.engine import Layer

import theano as T
import theano.tensor as TS
from theano.printing import Print


class HRNN_encoder(Layer):
    def __init__(self, input_dim, hidden_dim, FK_dim, depth, dropout_W, dropout_U,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 **kwargs):
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
        self.FK_dim = FK_dim
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.supports_masking = True
        self.gamma_init = initializations.get('one')
        self.beta_init = initializations.get('zero')
        self.epsilon = 1e-5

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(HRNN_encoder, self).__init__(**kwargs)



    def build(self, input_shape):

        self.W = self.init((self.input_dim+self.hidden_dim, self.hidden_dim), name='{}_W'.format(self.name))
        self.U = self.inner_init((self.input_dim+self.hidden_dim, self.hidden_dim), name='{}_U'.format(self.name))
        self.b = K.zeros((self.hidden_dim), name='{}_b'.format(self.name))


        self.W_FK_1 = self.init((self.input_dim+self.hidden_dim, self.FK_dim), name='{}_W_FK_1'.format(self.name))
        self.U_FK_1 = self.inner_init((self.input_dim+self.hidden_dim, self.FK_dim), name='{}_U_FK_1'.format(self.name))
        self.b_FK_1 = K.zeros((self.FK_dim,), name='{}_b_FK_1'.format(self.name))
        self.action_FK_right = K.zeros((self.FK_dim, ), name='{}_ACTION_FK_right'.format(self.name))
        self.action_FK_top = K.zeros((self.FK_dim, ), name='{}_ACTION_FK_top'.format(self.name))

        self.W_FK_2 = self.init((self.FK_dim, 1), name='{}_W_FK_2'.format(self.name))
        self.b_FK_2 = K.zeros((1,), name='{}_b_FK_2'.format(self.name))


        self.gammas = K.ones((2, self.hidden_dim), name="gammas")
        self.betas = K.zeros((2, self.hidden_dim), name="betas")
        self.trainable_weights = []
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return [None, None, None, None, None]

    def get_output_shape_for(self, input_shape):
        return [input_shape, (input_shape[0][0], self.hidden_dim), (self.depth, self.depth, input_shape[0][0], self.hidden_dim), (self.depth, self.depth, 1), (self.depth, self.depth, 1)]


    def call(self, input, mask=None):
        x = input[0]
        # Keras doesn't allow 1D model inputs - so that tensor have shape (1,1) instead of scalar or (1,)
        bucket_size = input[1][0][0]

        data_mask = mask[0]
        if data_mask.ndim == x.ndim-1:
            data_mask = K.expand_dims(data_mask)
        assert data_mask.ndim == x.ndim
        data_mask = data_mask.dimshuffle([1,0])

        ''' Pad input with zeros behind.
            Hidden vectors need to store info about input data and intermediate representations.
            So they are stored in vector independently. Length of vector = input_dim+hidden_dim;
            Input data transformed into (<input_dim> 00000000),
            result of each step - into (00000 <output_dim>)'''
        pad = K.zeros_like(x)
        pad = K.sum(pad, axis=(2))
        pad = K.expand_dims(pad)
        pad = K.tile(pad, (1, self.hidden_dim))
        x = K.concatenate([x, pad])

        batch_size = K.sum(K.zeros_like(x), axis=(1,2))
        initial_hor_fk = K.expand_dims(batch_size)
        initial_hor_fk = TS.unbroadcast(initial_hor_fk, 0, 1)

        x = x.dimshuffle([1,0,2])
        x = x[:bucket_size]
        x = TS.unbroadcast(x, 0,1,2)
        initial_fk = K.zeros_like(x)
        initial_fk = K.sum(initial_fk, axis=(2))
        initial_fk = K.expand_dims(initial_fk)
        initial_fk = TS.unbroadcast(initial_fk, 0, 1, 2)
        initial_has_value = K.ones_like(initial_fk)
        initial_hor_h = K.zeros_like(x[0])
        initial_hor_h = TS.unbroadcast(initial_hor_h, 0, 1)
        initial_hor_has_value = K.zeros_like(initial_hor_fk)
        data_mask = data_mask[:bucket_size]



        first_mask = K.zeros_like(data_mask[0])
        first_mask = K.expand_dims(first_mask, 0)
        mask2 = K.concatenate([data_mask[1:], first_mask], axis=0)
        mask2 = data_mask*(1-mask2)
        mask2 = K.expand_dims(mask2)
        #mask2 = K.concatenate([first_mask, mask2], axis=0)
        #mask2 = 1, if that sentence is over. That param required for making FK = 0 at the end of each sentence
        batch_zeros = TS.zeros_like(batch_size)
        batch_ones = TS.ones_like(batch_size)
        mask3 = K.expand_dims(batch_zeros)
        mask3 = K.repeat(mask3, self.depth-1)
        batch_ones = K.expand_dims(batch_ones)
        batch_ones = K.expand_dims(batch_ones)
        mask3 = K.concatenate([mask3, batch_ones], axis=1)
        mask3 = mask3.dimshuffle([1,0,2])
        data_mask = K.expand_dims(data_mask)
        initial_fk_calculated = K.zeros_like(initial_fk)
        initial_hor_fk_calculated = K.zeros_like(initial_hor_fk)

        results, _ = T.scan(self.vertical_step,
                            sequences=[mask3],
                            outputs_info=[x, initial_fk, initial_has_value, initial_fk_calculated],
                            non_sequences=[bucket_size, initial_hor_h, initial_hor_fk, data_mask, mask2, initial_hor_has_value, initial_hor_fk_calculated],
                            n_steps=self.depth)
        output = results[0][-1, -1, :, self.input_dim:]
        all_h = results[0].dimshuffle([2,0,1,3])
        fk = results[1].dimshuffle([2,0,1,3])
        has_value = results[2].dimshuffle([2,0,1,3])
        fk_calculated = results[3].dimshuffle([2,0,1,3])

        return [input[0], output, all_h, fk, fk_calculated]

    # Vertical pass along hierarchy dimension
    def vertical_step(self, *args):
        mask3 = args[0]
        x = args[1]
        fk_prev = args[2]
        has_value_prev = args[3]
        initial_fk_calculated = args[4]
        bucket_size=args[5]
        initial_h = args[6]
        initial_fk=args[7]
        mask = args[8]
        mask2 = args[9]
        initial_has_value = args[10]
        initial_hor_fk_calculated = args[11]

        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, fk_prev, mask, mask2, has_value_prev],
                            outputs_info=[initial_h, initial_fk, initial_has_value, initial_hor_fk_calculated],
                            non_sequences=[mask3],
                            n_steps=bucket_size)
        h = results[0]
        fk = results[1]
        has_value = results[2]
        fk_calculated = results[3]

        #Shift computed FK for 1 step left because at the step i we compute FK for i-1
        last_fk = K.zeros_like(fk[0])
        last_fk = K.expand_dims(last_fk, 0)
        shifted_fk = K.concatenate([fk[1:], last_fk], axis=0)
        shifted_fk = TS.unbroadcast(shifted_fk, 0, 1)
        # Uncomment to monitor FK values during testing
        #shifted_fk = Print("shifted_fk")(shifted_fk)
        #has_value = Print("has_value")(has_value)


        return h, shifted_fk, has_value, fk_calculated

    # Horizontal pass along time dimension
    def horizontal_step(self, *args):
        x = args[0]
        fk_prev = args[1]
        mask = args[2]
        mask2 = args[3]
        has_value_prev = args[4]
        h_tm1 = args[5]
        fk_tm1 = args[6]
        has_value_tm1 = args[7]
        fk_calculated_tm1 = args[8]
        mask3 = args[9]


        if 0 < self.dropout_U < 1:
            ones = K.ones((self.input_dim+self.hidden_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
        else:
            B_U = K.cast_to_floatx(1.)
        if 0 < self.dropout_W < 1:
            ones = K.ones((self.input_dim+self.hidden_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
        else:
            B_W = K.cast_to_floatx(1.)


        sum1_fk = K.dot(x, self.W_FK_1)
        sum2_fk = K.dot(h_tm1, self.U_FK_1)
        sum_fk = sum1_fk + sum2_fk + self.b_FK_1

        fk_candidate_right_1 = self.inner_activation(sum_fk+self.action_FK_right)
        fk_candidate_top_1 = self.inner_activation(sum_fk+self.action_FK_top)

        fk_candidate_right_2 = self.inner_activation(K.dot(fk_candidate_right_1,self.W_FK_2)+self.b_FK_2)
        fk_candidate_top_2 = self.inner_activation(K.dot(fk_candidate_top_1,self.W_FK_2)+self.b_FK_2)

        fk_candidate = K.switch(TS.le(fk_candidate_right_2, fk_candidate_top_2), 1, 0)

        fk = fk_prev + (1-fk_prev)*fk_candidate
        fk = K.switch(mask3, 1, fk)
        fk = K.switch(mask2, 0, fk)

        fk_calculated = mask*(1-mask3)*(1-mask2)*(1-fk_prev)

        fk_prev_expanded = K.repeat_elements(fk_prev, self.hidden_dim, 1)
        fk_expanded = K.repeat_elements(fk, self.hidden_dim, 1)

        sum1 = (1-fk_prev_expanded)*self.ln(K.dot(x*B_W, self.W), self.gammas[0], self.betas[0])
        sum2 = fk_expanded*self.ln(K.dot(h_tm1*B_U, self.U), self.gammas[1], self.betas[1])

        # Actual new hidden state if node got info from left and from below
        sum = sum1 + sum2 + self.b
        h_ = self.activation(sum)

        # Pad with zeros in front
        zeros = K.zeros_like(h_)
        zeros = K.sum(zeros, axis=(1))
        zeros = K.expand_dims(zeros)
        pad = K.tile(zeros, (1, self.input_dim))
        h_ = K.concatenate([pad, h_])

        ''' Total formula for h and FK
            If information flows from one direction only then simply propagate that value further
            (direction for propagation will be calculated on the next step)
            Otherwise - perform default recurrent computation
        '''

        h_tm1_only = has_value_tm1*fk*(fk_prev+(1-fk_prev)*(1-has_value_prev))
        x_only = has_value_prev*(1-fk_prev)*((1-fk)+fk*(1-has_value_tm1))
        both = (1-fk_prev)*fk*has_value_tm1*has_value_prev

        h_tm1_only_expanded = K.repeat_elements(h_tm1_only, self.hidden_dim+self.input_dim, 1)
        x_only_expanded = K.repeat_elements(x_only, self.hidden_dim+self.input_dim, 1)
        both_expanded = K.repeat_elements(both, self.hidden_dim+self.input_dim, 1)

        h = h_tm1_only_expanded*h_tm1 + x_only_expanded*x + both_expanded*h_
        has_value = h_tm1_only + x_only + both

        # Apply mask
        h = K.switch(mask, h, h_tm1)
        fk = K.switch(mask, fk, fk_tm1)
        has_value = K.switch(mask, has_value, has_value_tm1)

        result = [h, fk, has_value, fk_calculated]
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
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(HRNN_encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))