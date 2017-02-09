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
        В слое применяются:
        * Layer Normalization
        * Bucketing (размер корзины подается на вход в качестве тензора (1,1) вместе с основным тензором
        * Masking
        * Dropout

        :param input_dim: размерность входных векторов (символы\слова)
        :param hidden_dim: размерность внутренних и выходных векторов слоя
        :param depth: глубина иерархии
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
        self.W = self.init((self.input_dim+self.hidden_dim, 2*self.hidden_dim+self.input_dim), name='{}_W'.format(self.name))
        self.U = self.inner_init((self.input_dim+self.hidden_dim, 2*self.hidden_dim+self.input_dim), name='{}_U'.format(self.name))
        self.b = K.zeros((2*self.hidden_dim+self.input_dim), name='{}_b'.format(self.name))
        self.gammas = K.ones((2, 2*self.hidden_dim+self.input_dim), name="gammas")
        self.betas = K.zeros((2, 2*self.hidden_dim+self.input_dim), name="betas")
        self.trainable_weights = [self.W ,self.U , self.b, self.gammas, self.betas]
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], self.hidden_dim)


    def call(self, input, mask=None):
        x = input[0]
        # Keras не позволяет одномерные входы для моделей - поэтому тензор формы (1,1), а не скаляр или (1,)
        bucket_size = input[1][0][0]

        data_mask = mask[0]
        if data_mask.ndim == x.ndim-1:
            data_mask = K.expand_dims(data_mask)
        assert data_mask.ndim == x.ndim
        data_mask = data_mask.dimshuffle([1,0])

        ''' Дополняем нулями сзади.
            Входные тензоры для сети должны уметь содержать в себе информацию и о исходных данных (символы\слова),
            и о результатах работы сети.
            Поэтому длина тензора = input_dim+hidden_dim; исходные данные превращаются в (<input_dim> 00000000),
            а результат работы каждого шага сети - в (00000 <output_dim>)'''
        pad = K.zeros_like(x)
        pad = K.sum(pad, axis=(2))
        pad = K.expand_dims(pad)
        pad = K.tile(pad, (1, self.hidden_dim))
        x = K.concatenate([x, pad])

        initial_hor_fk = K.sum(K.zeros_like(x), axis=(1,2))
        initial_hor_fk = K.expand_dims(initial_hor_fk)
        initial_hor_fk = K.tile(initial_hor_fk, (1, self.input_dim+self.hidden_dim))

        x = x.dimshuffle([1,0,2])
        x = x[:bucket_size]
        x = TS.unbroadcast(x, 0,1,2)
        initial_fk = K.zeros_like(x)
        initial_fk = K.sum(initial_fk, axis=(2)) #(max_len,samples)
        initial_fk = K.expand_dims(initial_fk)
        initial_fk = K.tile(initial_fk, (1, self.input_dim+self.hidden_dim))
        initial_fk = TS.unbroadcast(initial_fk, 0, 1)
        initial_hor_h = K.zeros_like(x[0]) #(samples, emb_size)
        initial_hor_h = TS.unbroadcast(initial_hor_h, 0, 1)
        data_mask = data_mask[:bucket_size]

        results, _ = T.scan(self.vertical_step,
                            sequences=[],
                            outputs_info=[x, initial_fk],
                            non_sequences=[bucket_size, initial_hor_h, initial_hor_fk, data_mask],
                            n_steps=self.depth)
        outputs = results[0]
        outputs = outputs[-1,-1,:,self.input_dim:]
        return outputs

    # Проход в вертикальном направлении - вдоль измерения иерархии
    def vertical_step(self, *args):
        x = args[0]
        fk_prev = args[1]
        bucket_size=args[2]
        initial_h = args[3]
        initial_fk=args[4]
        mask = args[5]

        first_mask = K.zeros_like(mask[0])
        first_mask = K.expand_dims(first_mask, 0)
        mask2 = K.concatenate([mask[1:], first_mask], axis=0)
        mask2 = mask*(1-mask2)
        mask2 = K.expand_dims(mask2)
        #mask2 = 1, если строка закончилась. Этот параметр нужен, т.к. на последнем шаге информационный поток надо всегда отдавать наверх

        results, _ = T.scan(self.horizontal_step,
                            sequences=[x, fk_prev, mask, mask2],
                            outputs_info=[initial_h, initial_fk],
                            n_steps=bucket_size)
        h = results[0]
        fk = results[1]

        #Сдвигаем рассчитанные fk на один шаг влево - т.к. на шаге i мы считаем fk для предыдущего шага
        last_fk = K.zeros_like(fk[0])
        last_fk = K.expand_dims(last_fk, 0)
        shifted_fk = K.concatenate([fk[1:], last_fk], axis=0)
        shifted_fk = TS.unbroadcast(shifted_fk, 0, 1)
        #shifted_fk = Print("shifted_fk")(shifted_fk)

        return h, shifted_fk

    # Проход в горизонтальном направлении - вдоль измерения времени
    def horizontal_step(self, *args):
        x = args[0]
        fk_prev = args[1]
        mask = args[2]
        mask2 = args[3]
        h_tm1 = args[4]
        fk_tm1 = args[5]

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
        total_sum = sum1 + sum2 + self.b

        fk_candidate = self.inner_activation(total_sum[:, :self.input_dim+self.hidden_dim])

        # Фактическое новое состояние сети, если информация пришла слева и снизу. Учет FK - в итоговой сумме
        h_ = self.activation(total_sum[:, self.input_dim+self.hidden_dim:])

        # Дополняем нулями спереди
        zeros = K.zeros_like(h_)
        zeros = K.sum(zeros, axis=(1))
        zeros = K.expand_dims(zeros)
        pad = K.tile(zeros, (1, self.input_dim))
        h_ = K.concatenate([pad, h_])


        ''' Итоговая формула h
            Если информация приходит только с одного из направлений - то просто передаем это значение дальше
            (направление, в котором передавать, будет вычислено на следующем шаге)
            Иначе - выполняем вычисление по стандартной рекурсивной формуле
        '''
        h_candidate = (1-fk_candidate)*x + fk_candidate*h_
        h = fk_prev*h_tm1 + (1-fk_prev)*h_candidate
        fk = fk_prev + (1-fk_prev)*fk_candidate


        mask_for_h = K.expand_dims(mask)
        # Если это последний элемент в последовательности, то всегда отдаем его наверх
        fk = K.switch(mask2, 0, fk)
        # Применяем маску - пропускаем оставшиеся 0 от фактического конца строки до конца батча
        output1 = K.switch(mask_for_h, h, h_tm1)
        output2 = K.switch(mask_for_h, fk, fk_tm1)
        result = [output1, output2]
        return result


    # Формула для Linear Normalization
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