import keras.backend as K
from keras.layers import initializations, activations
from keras.engine import Layer

import theano as T
import theano.tensor as TS
from theano.printing import Print


class Predictor(Layer):
    def __init__(self, input_dim, hidden_dim, action_dim, depth, batch_size, max_len,
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
        self.depth = depth
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_len = max_len
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.supports_masking = True
        self.gamma_init = initializations.get('one')
        self.beta_init = initializations.get('zero')
        self.epsilon = 1e-5
        super(Predictor, self).__init__(**kwargs)



    def build(self, input_shape):
        self.W_emb = self.init((self.input_dim, self.hidden_dim), name='{}_W_emb'.format(self.name))
        self.b_emb = K.zeros((self.hidden_dim), name='{}_b_emb'.format(self.name))

        self.W = self.init((self.hidden_dim, 3*self.hidden_dim), name='{}_W'.format(self.name))
        self.U = self.inner_init((self.hidden_dim, 3*self.hidden_dim), name='{}_U'.format(self.name))
        self.b = K.zeros((3*self.hidden_dim), name='{}_b'.format(self.name))


        self.W_action_1 = self.init((self.hidden_dim, self.action_dim), name='{}_W_action_1'.format(self.name))
        self.U_action_1 = self.inner_init((self.hidden_dim, self.action_dim), name='{}_U_action_1'.format(self.name))
        self.b_action_1 = K.zeros((self.action_dim,), name='{}_b_action_1'.format(self.name))

        self.W_action_2 = self.init((self.action_dim,2), name='{}_W_action_2'.format(self.name))
        self.b_action_2 = K.zeros((2,), name='{}_b_action_2'.format(self.name))


        self.gammas = K.ones((2, self.hidden_dim), name="gammas")
        self.betas = K.zeros((2, self.hidden_dim), name="betas")
        self.trainable_weights = []
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return [None, None, None, None, None, None]

    def get_output_shape_for(self, input_shape):
        return [(input_shape[0][0], self.hidden_dim),
                (self.batch_size, self.depth, 1, 1),
                (self.batch_size, self.depth, 1, 1),
                (self.batch_size, self.depth, 1,self.hidden_dim),
                (self.batch_size, self.depth, 1,self.hidden_dim),
                (self.batch_size, self.depth, 1,2)]


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
        initial_action_calculated = TS.zeros_like(x[:,:,0], dtype="int8")
        first_mask = K.zeros_like(data_mask[0])
        first_mask = K.expand_dims(first_mask, 0)
        eos_mask = K.concatenate([data_mask[1:], first_mask], axis=0)
        eos_mask = TS.cast(data_mask*(1-eos_mask), "int8")
        last_layer_mask3 = K.concatenate([K.zeros((self.depth-1,self.batch_size), dtype="int8"), K.ones((1, self.batch_size), dtype="int8")], axis=0)
        initial_policy = K.zeros_like(x[:,:,0])
        initial_policy = initial_policy.dimshuffle([0,1,'x'])
        initial_policy = TS.extra_ops.repeat(initial_policy, 2, axis=2)
        initial_policy_input_x = K.zeros_like(x)
        initial_policy_input_h = K.zeros_like(x)


        results, _ = T.scan(self.vertical_step,
                            sequences=[last_layer_mask3],
                            outputs_info=[x, initial_action, data_mask, initial_action_calculated, initial_policy_input_x, initial_policy_input_h, initial_policy],
                            non_sequences=[bucket_size, eos_mask],
                            n_steps=self.depth)
        output = results[0][-1, -1, :, :]
        action = results[1].dimshuffle([2,0,1])
        action_calculated = results[3].dimshuffle([2,0,1])
        policy_input_x = results[4].dimshuffle([2,0,1,3])
        policy_input_h = results[5].dimshuffle([2,0,1,3])
        policy = results[6].dimshuffle([2,0,1,3])

        return [output, action, action_calculated, policy_input_x, policy_input_h, policy]

    # Vertical pass along hierarchy dimension
    def vertical_step(self, *args):
        last_layer_mask3 = args[0]
        x = args[1]
        action_prev = args[2]
        data_mask = args[3]
        action_calculated_prev = args[4]
        policy_input_x_prev = args[5]
        policy_input_h_prev = args[6]
        policy_prev = args[7]
        bucket_size=args[8]
        eos_mask = args[9]

        initial_h = K.zeros((self.batch_size, self.hidden_dim))
        initial_action = K.zeros((self.batch_size), dtype="int8")
        initial_data_mask = K.zeros((self.batch_size), dtype="int8")
        initial_action_calculated = K.zeros((self.batch_size), dtype="int8")
        initial_policy = K.zeros((self.batch_size, 2))

        shifted_data_mask = K.concatenate([K.ones((1, self.batch_size)), data_mask[:-1]], axis=0)
        shifted_eos_mask = K.concatenate([K.zeros((1, self.batch_size)), eos_mask[:-1]], axis=0)



        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, action_prev, shifted_data_mask, data_mask, shifted_eos_mask, eos_mask],
                            outputs_info=[initial_h, initial_action, initial_data_mask, initial_action_calculated, initial_policy],
                            non_sequences=[last_layer_mask3],
                            n_steps=bucket_size)
        h = results[0]
        action = results[1]
        new_data_mask = results[2]
        action_calculated = results[3]
        policy = results[4]

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

        return h, shifted_action, new_data_mask, action_calculated, policy_input_x, policy_input_h, policy

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
        action_calculated_tm1 = args[9]
        policy_tm1 = args[10]
        last_layer_mask3 = args[11]


        policy = activations.relu(K.dot(x, self.W_action_1) + K.dot(h_tm1, self.U_action_1) + self.b_action_1)
        policy = TS.exp(K.dot(policy, self.W_action_2)+self.b_action_2)

        action = K.switch(TS.ge(policy[:,0], policy[:, 1]), 1, 0)
        action = K.switch(action_prev, 1, action)
        action = K.switch(last_layer_mask3, 1, action)
        action = K.switch(eos_mask_tm1, 0, action)
        action = TS.cast(action, "int8")

        action_calculated = data_mask_prev_tm1*(1-last_layer_mask3)*(1-eos_mask_tm1)*(1-action_prev)
        action_calculated = TS.cast(action_calculated, "int8")

        # Actual new hidden state if node got info from left and from below
        z = K.hard_sigmoid(K.dot(x, self.W[:,:self.hidden_dim]) + K.dot(h_tm1, self.U[:,:self.hidden_dim]) + self.b[:self.hidden_dim])
        r = K.hard_sigmoid(K.dot(x, self.W[:,self.hidden_dim:2*self.hidden_dim]) + K.dot(h_tm1, self.U[:,self.hidden_dim:2*self.hidden_dim]) + self.b[self.hidden_dim:2*self.hidden_dim])
        h_ = z*h_tm1 + (1-z)*K.tanh(K.dot(x, self.W[:,2*self.hidden_dim:]) + K.dot(r*h_tm1, self.U[:,2*self.hidden_dim:]) + self.b[2*self.hidden_dim:])

        zeros = K.zeros((self.batch_size, self.hidden_dim))
        both = (1-action_prev)*data_mask_prev*action*data_mask_tm1
        h_tm1_only = data_mask_tm1*action*(action_prev + (1-action_prev)*(1-data_mask_prev))
        x_only = data_mask_prev*(1-action_prev)*((1-action) + action*(1-data_mask_tm1))

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
        data_mask = data_mask_prev*data_mask
        data_mask_prev = data_mask_prev.dimshuffle([0,'x'])
        data_mask_prev = TS.extra_ops.repeat(data_mask_prev, self.hidden_dim, axis=1)
        h = K.switch(data_mask_prev, h, h_tm1)

        result = [h, action, data_mask, action_calculated, policy]
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
        base_config = super(Predictor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))