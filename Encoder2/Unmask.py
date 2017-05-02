# -*- coding: utf-8 -*-
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer



from keras.initializers import glorot_uniform, orthogonal, zeros
import theano as T
import theano.tensor as TS
from theano.printing import Print

class Unmask(Layer):
    def compute_mask(self, input, input_mask=None):
        return input[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, input, mask=None):
        return input[0]
