import keras.backend as K
from keras.layers import initializations, activations
from keras.engine import Layer

import theano as T
import theano.tensor as TS
from theano.printing import Print


class HRNN_encoder(Layer):
    def __init__(self, input_dim, hidden_dim, depth, dropout_W, dropout_U,
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
        self.W = self.init((self.input_dim+self.hidden_dim, self.hidden_dim+1), name='{}_W'.format(self.name))
        self.U = self.inner_init((self.input_dim+self.hidden_dim, self.hidden_dim+1), name='{}_U'.format(self.name))
        self.b = K.zeros((self.hidden_dim+1), name='{}_b'.format(self.name))
        self.gammas = K.ones((2, self.hidden_dim+1), name="gammas")
        self.betas = K.zeros((2, self.hidden_dim+1), name="betas")
        self.trainable_weights = [self.W ,self.U , self.b, self.gammas, self.betas]
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

        initial_hor_fk = K.sum(K.zeros_like(x), axis=(1,2))

        x = x.dimshuffle([1,0,2])
        x = x[:bucket_size]
        x = TS.unbroadcast(x, 0,1,2)
        initial_fk = K.zeros_like(x)
        initial_fk = K.sum(initial_fk, axis=(2))
        initial_fk = TS.unbroadcast(initial_fk, 0, 1)
        initial_has_value = K.ones_like(initial_fk)
        initial_hor_h = K.zeros_like(x[0])
        initial_hor_h = TS.unbroadcast(initial_hor_h, 0, 1)
        initial_hor_has_value = K.zeros_like(initial_hor_fk)
        data_mask = data_mask[:bucket_size]


        first_mask = K.zeros_like(data_mask[0])
        first_mask = K.expand_dims(first_mask, 0)
        mask2 = K.concatenate([data_mask[1:], first_mask], axis=0)
        mask2 = data_mask*(1-mask2)
        mask2 = K.concatenate([first_mask, mask2], axis=0)
        #mask2 = 1, if that sentence is over. That param required for making FK = 0 at the end of each sentence

        results, _ = T.scan(self.vertical_step,
                            sequences=[],
                            outputs_info=[x, initial_fk, initial_fk, initial_has_value],
                            non_sequences=[bucket_size, initial_hor_h, initial_hor_fk, data_mask, mask2, initial_hor_has_value],
                            n_steps=self.depth)
        outputs = results[0]
        outputs = outputs[-1,-1,:,self.input_dim:]
        return outputs

    # Vertical pass along hierarchy dimension
    def vertical_step(self, *args):
        x = args[0]
        fk_prev_tm1 = args[1]
        fk_prev = args[2]
        has_value_prev = args[3]
        bucket_size=args[4]
        initial_h = args[5]
        initial_fk=args[6]
        mask = args[7]
        mask2 = args[8]
        initial_has_value = args[9]

        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, fk_prev_tm1, fk_prev, mask, mask2, has_value_prev],
                            outputs_info=[initial_h, initial_fk, initial_has_value],
                            n_steps=bucket_size)
        h = results[0]
        fk = results[1]
        has_value = results[2]

        #Shift computed FK for 1 step left because at the step i we compute FK for i-1
        last_fk = K.zeros_like(fk[0])
        last_fk = K.expand_dims(last_fk, 0)
        shifted_fk = K.concatenate([fk[1:], last_fk], axis=0)
        shifted_fk = TS.unbroadcast(shifted_fk, 0, 1)
        # Uncomment to monitor FK values during testing
        #shifted_fk = Print("shifted_fk")(shifted_fk)
        #has_value = Print("has_value")(has_value)


        return h, fk, shifted_fk, has_value

    # Horizontal pass along time dimension
    def horizontal_step(self, *args):
        x = args[0]
        fk_prev_tm1 = args[1]
        fk_prev = args[2]
        mask = args[3]
        mask2 = args[4]
        has_value_prev = args[5]
        h_tm1 = args[6]
        fk_tm1 = args[7]
        has_value_tm1 = args[8]


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

        sum1 = self.ln(K.dot(x*B_W, self.W), self.gammas[0], self.betas[0])
        sum2 = self.ln(K.dot(h_tm1*B_U, self.U), self.gammas[1], self.betas[1])
        sum = sum1 + sum2 + self.b


        fk_candidate_both = self.inner_activation(sum[:, 0])
        fk_candidate_tm1 = self.inner_activation((sum2+self.b)[:, 0])

        fk = fk_prev_tm1 + (1-fk_prev_tm1)*(fk_tm1*fk_candidate_both+(1-fk_tm1)*fk_candidate_tm1)
        fk = K.switch(mask2, 0, fk)
        #fk = Print("fk")(fk)


        # Actual new hidden state if node got info from left and from below
        h_ = self.activation(sum[:, 1:])

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

        #h_tm1_only = Print("h_tm1_only")(h_tm1_only)
        #x_only = Print("x_only")(x_only)
        #both = Print("both")(both)

        h_tm1_only_expanded = K.expand_dims(h_tm1_only)
        h_tm1_only_expanded = K.repeat_elements(h_tm1_only_expanded, self.hidden_dim+self.input_dim, 1)
        x_only_expanded = K.expand_dims(x_only)
        x_only_expanded = K.repeat_elements(x_only_expanded, self.hidden_dim+self.input_dim, 1)
        both_expanded = K.expand_dims(both)
        both_expanded = K.repeat_elements(both_expanded, self.hidden_dim+self.input_dim, 1)

        h = h_tm1_only_expanded*h_tm1 + x_only_expanded*x + both_expanded*h_
        has_value = 1 - (1-h_tm1_only)*(1-x_only)*(1-both)

        mask_for_h = K.expand_dims(mask)
        # Apply mask
        h = K.switch(mask_for_h, h, h_tm1)
        fk = K.switch(mask, fk, fk_tm1)
        has_value = K.switch(mask, has_value, has_value_tm1)
        # Make FK = 0 if that's last element of sequence
        fk = K.switch(mask2, 0, fk)

        result = [h, fk, has_value]
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