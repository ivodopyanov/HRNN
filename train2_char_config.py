# -*- coding: utf-8 -*-

def init_settings():
    result = {}
    result['max_features'] = 20000
    result['batch_size']=8
    result['max_len'] = 64
    result['epoch_mult']=1
    result['char_units_ep']=128
    result['char_units']=128
    result['word_units']=128
    result['epochs']=100
    result['dropout_u']=0
    result['dropout_w']=0
    result['l2']=1e-5
    result['hidden'] = 128
    return result