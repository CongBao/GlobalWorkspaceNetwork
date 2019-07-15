# model.py
# author: Cong Bao

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
import copy
import json
import math

import numpy as np
import tensorflow as tf



class GWTConfig(object):

    def __init__(self):
        self.gws_size = 64
        self.proj_size = 64
        self.inter_size = 128
        self.hidden_size = 32
        self.n_head = 2
        self.self_atten = True
        self.atten_type = 'general'
        self.map_activ = None
        self.proj_activ = 'tanh'
        self.inter_activ = 'gelu'
        self.drop_rate = 0.1



class GWTModel(object):

    def __init__(self, inputs, config, is_training):

        config = copy.deepcopy(config)
        if not is_training:
            config.drop_rate = 0.0

        with tf.variable_scope('gwt_model'):
            with tf.variable_scope('mapping'): # input -> dense (sequence)
                self.mapped_inputs = mapping(
                    input_list=inputs,
                    units=config.hidden_size,
                    activ=get_activ_fn(config.map_activ),
                    dropout=config.drop_rate
                ) # M, (B, S, F) -> (B, M, S, H)
            with tf.variable_scope('gws'): # dense output -> rnn cell (loop)
                self.outputs, self.dists = global_workspace(
                    inputs=self.mapped_inputs,
                    gws_size=config.gws_size,
                    n_head=config.n_head,
                    head_size=config.hidden_size,
                    inter_size=config.inter_size,
                    inter_activ=get_activ_fn(config.inter_activ),
                    atten_type=config.atten_type,
                    drop_rate=config.drop_rate,
                    self_atten=config.self_atten
                ) # (B, M, S, H) -> (B, S, G), (B, S, N, [1, M], M)
            with tf.variable_scope('projection'): # rnn last cell state -> fc
                self.proj_output = projection(
                    features=self.outputs[:, -1, :],
                    units=config.proj_size,
                    activ=get_activ_fn(config.proj_activ),
                    dropout=config.drop_rate
                ) # (B, G) -> (B, P)

    def get_projection(self):
        return self.proj_output

    def get_dist_sequence(self):
        return self.dists

    def get_gws_sequence(self):
        return self.outputs



class ConcModel(object):

    def __init__(self, inputs, config, is_training):

        config = copy.deepcopy(config)
        if not is_training:
            config.drop_rate = 0.0

        with tf.variable_scope('conc_model'):
            self.outputs, _ = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.LSTMCell(config.gws_size),
                inputs=tf.concat(inputs, axis=-1), # M, (B, S, F) -> (B, S, F*)
                dtype=tf.float32
            )
            with tf.variable_scope('projection'):
                self.proj_output = projection(
                    features=self.outputs[:,-1,:],
                    units=config.proj_size,
                    activ=get_activ_fn(config.proj_activ),
                    dropout=config.drop_rate
                ) # (B, G) -> (B, P)

    def get_projection(self):
        return self.proj_output

    def get_gws_sequence(self):
        return self.outputs



class MapConcModel(object):

    def __init__(self, inputs, config, is_training):

        config = copy.deepcopy(config)
        if not is_training:
            config.drop_rate = 0.0

        with tf.variable_scope('map_conc_model'):
            with tf.variable_scope('mapping'): # input -> dense (sequence)
                self.mapped_inputs = mapping(
                    input_list=inputs,
                    units=config.hidden_size,
                    activ=get_activ_fn(config.map_activ),
                    dropout=config.drop_rate
                ) # M, (B, S, F) -> (B, M, S, H)
            input_list = tf.unstack(self.mapped_inputs, axis=1) # (B, M, S, H) -> M, (B, S, H)
            inputs = tf.concat(input_list, axis=-1) # M, (B, S, H) -> (B, S, M*H)
            self.outputs, _ = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.LSTMCell(config.gws_size),
                inputs=inputs,
                dtype=tf.float32
            )
            with tf.variable_scope('projection'):
                self.proj_output = projection(
                    features=self.outputs[:,-1,:],
                    units=config.proj_size,
                    activ=get_activ_fn(config.proj_activ),
                    dropout=config.drop_rate
                ) # (B, G) -> (B, P)

    def get_projection(self):
        return self.proj_output

    def get_gws_sequence(self):
        return self.outputs



# scalar dimensions reference
#  B = batch size
#  M = number of modalities
#  S = sequence length
#  F = feature size before mapping
#  G = global workspace size
#  N = number of heads
#  H = hidden size (size of each head)
#  I = intermediate size
#  P = projection size

# scalar dimensions for general attention
# SF = subject tensor feature size
# OF = object tensor feature size
# SL = subject tensor sequence length
# OL = object tensor sequence length

def mapping(input_list, units, activ=None, dropout=0.0):

    n_modality = len(input_list) # M

    mapped = []
    for idx in range(n_modality):
        features = input_list[idx] # (B, S, F)
        feat_shape = get_shape(features, expected_rank=3)
        feats = tf.nn.dropout(
            x=features,
            rate=dropout,
            noise_shape=(feat_shape[0], 1, feat_shape[2])
        ) # (B, S, F)
        feats = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dense(units, activation=activ),
            name='map_{}'.format(idx)
        )(feats) # (B, S, H)
        mapped.append(feats)

    inputs = tf.stack(mapped) # (M, B, S, H)
    inputs = tf.transpose(inputs, [1, 0, 2, 3]) # (B, M, S, H)
    
    return inputs



def projection(features, units, activ=None, dropout=0.0):
    
    assert_rank(features, 2) # (B, G)

    proj = tf.nn.dropout(features, rate=dropout)
    proj = tf.keras.layers.Dense(
        units=units,
        activation=activ
    )(proj) # (B, P)

    return proj



def attention(sbj, obj, n_head, head_size, inter_size, inter_activ, atten_type, drop_rate):

    sbj_shape = get_shape(sbj, expected_rank=3) # (B, SL, SF) : (B, 1, G) or (B, M, H)
    obj_shape = get_shape(obj, expected_rank=3) # (B, OL, OF) : (B, M, H)

    batch_size = sbj_shape[0] # B
    sbj_length = sbj_shape[1] # SL : 1 or M
    obj_length = obj_shape[1] # OL : M
    sbj_feats = sbj_shape[2] # SF : G or H
    obj_feats = obj_shape[2] # OF : H

    def general(sbj, obj, n_head, head_size):
        """ General (multiplicative) attention. (Luong 2015) """
        sbj_tensors = tf.reshape(sbj, [-1, sbj_feats]) # (B*SL, SF)
        obj_tensors = tf.reshape(obj, [-1, obj_feats]) # (B*OL, OF)
        key = tf.keras.layers.Dense(n_head*sbj_feats, name='key')(obj_tensors) # (B*OL, N*SF)
        key = tf.reshape(key, [batch_size, obj_length, n_head, sbj_feats]) # (B, OL, N, SF)
        key = tf.transpose(key, [0, 2, 1, 3]) # (B, N, OL, SF)
        query = tf.reshape(sbj_tensors, [batch_size, 1, sbj_length, sbj_feats]) # (B, 1, SL, SF)
        query = tf.tile(query, [1, n_head, 1, 1]) # (B, N, SL, SF)
        scores = tf.matmul(query, key, transpose_b=True) # (B, N, SL, OL)
        dist = tf.nn.softmax(scores) # (B, N, SL, OL)
        return dist

    def additive(sbj, obj, n_head, head_size):
        """ Additive attention. (Bahdanau 2015) """
        sbj_tensors = tf.reshape(sbj, [-1, sbj_feats]) # (B*SL, SF)
        obj_tensors = tf.reshape(obj, [-1, obj_feats]) # (B*OL, OF)
        query = tf.keras.layers.Dense(n_head*head_size, name='query')(sbj_tensors) # (B*SL, N*H)
        query = tf.reshape(query, [batch_size, sbj_length, n_head, head_size]) # (B, SL, N, H)
        query = tf.transpose(query, [0, 2, 1, 3]) # (B, N, SL, H)
        query = tf.expand_dims(query, 3) # (B, N, SL, 1, H)
        query = tf.tile(query, [1, 1, 1, obj_length, 1]) # (B, N, SL, OL, H)
        key = tf.keras.layers.Dense(n_head*head_size, name='key')(obj_tensors) # (B*OL, N*H)
        key = tf.reshape(key, [batch_size, obj_length, n_head, head_size]) # (B, OL, N, H)
        key = tf.transpose(key, [0, 2, 1, 3]) # (B, N, OL, H)
        key = tf.expand_dims(key, 2) # (B, N, 1, OL, H)
        key = tf.tile(key, [1, 1, sbj_length, 1, 1]) # (B, N, SL, OL, H)
        add = tf.add(query, key) # (B, N, SL, OL, H)
        add = tf.nn.tanh(add) # (B, N, SL, OL, H)
        add = tf.reshape(add, [-1, head_size]) # (B*N*SL*OL, H)
        scores = tf.keras.layers.Dense(1, name='score')(add) # (B*N*SL*OL, 1)
        scores = tf.reshape(scores, [batch_size, n_head, sbj_length, obj_length]) # (B, N, SL, OL)
        dist = tf.nn.softmax(scores) # (B, N, SL, OL)
        return dist

    def scaled_dot_product(sbj, obj, n_head, head_size):
        """ Scaled dot product attention. (Vaswani 2017) """
        sbj_tensors = tf.reshape(sbj, [-1, sbj_feats]) # (B*SL, SF)
        obj_tensors = tf.reshape(obj, [-1, obj_feats]) # (B*OL, OF)
        query = tf.keras.layers.Dense(n_head*head_size, name='query')(sbj_tensors) # (B*SL, N*H)
        query = tf.reshape(query, [batch_size, sbj_length, n_head, head_size]) # (B, SL, N, H)
        query = tf.transpose(query, [0, 2, 1, 3]) # (B, N, SL, H)
        key = tf.keras.layers.Dense(n_head*head_size, name='key')(obj_tensors) # (B*OL, N*H)
        key = tf.reshape(key, [batch_size, obj_length, n_head, head_size]) # (B, OL, N, H)
        key = tf.transpose(key, [0, 2, 1, 3]) # (B, N, OL, H)
        scores = tf.matmul(query, key, transpose_b=True) # (B, N, SL, OL)
        scores = tf.multiply(scores, 1./math.sqrt(float(head_size))) # (B, N, SL, OL)
        dist = tf.nn.softmax(scores) # (B, N, SL, OL)
        return dist

    atten_dict = {
        'general': general,
        'additive': additive,
        'scaled_dot_product': scaled_dot_product
    }
    if atten_type not in atten_dict:
        raise ValueError('Unknown attention type: %s' % atten_type)

    with tf.variable_scope('attention'):
        obj_tensors = tf.reshape(obj, [-1, obj_feats]) # (B*OL, OF)
        with tf.variable_scope(atten_dict[atten_type].__name__):
            dist = atten_dict[atten_type](sbj, obj, n_head, head_size) # (B, N, SL, OL)
            value = tf.keras.layers.Dense(n_head*head_size, name='value')(obj_tensors) # (B*OL, N*H)
            value = tf.reshape(value, [batch_size, obj_length, n_head, head_size]) # (B, OL, N, H)
            value = tf.transpose(value, [0, 2, 1, 3]) # (B, N, OL, H)
            context = tf.matmul(dist, value) # (B, N, SL, H)
            context = tf.transpose(context, [0, 2, 1, 3]) # (B, SL, N, H)
            context = tf.reshape(context, [batch_size*sbj_length, n_head*head_size]) # (B*SL, N*H)
        with tf.variable_scope('atten_output'):
            atten_output = tf.keras.layers.Dense(head_size)(context) # (B*SL, H)
            if sbj_length != obj_length and sbj_length == 1:
                atten_output = tf.reshape(atten_output, [batch_size, sbj_length, head_size]) # (B, SL, H)
                atten_output = tf.tile(atten_output, [1, obj_length, 1]) # (B, OL, H)
                atten_output = tf.reshape(atten_output, [batch_size*obj_length, head_size]) # (B*OL, H)
            atten_output = tf.nn.dropout(atten_output, rate=drop_rate) # (B*OL, H)
            atten_output = tf.contrib.layers.layer_norm(obj_tensors + atten_output, begin_norm_axis=-1) # (B*OL, H)
        with tf.variable_scope('intermediate'):
            inter_output = tf.keras.layers.Dense(inter_size, inter_activ)(atten_output) # (B*OL, I)
        with tf.variable_scope('layer_output'):
            layer_output = tf.keras.layers.Dense(head_size)(inter_output) # (B*OL, H)
            layer_output = tf.nn.dropout(layer_output, rate=drop_rate) # (B*OL, H)
            layer_output = tf.contrib.layers.layer_norm(layer_output + atten_output, begin_norm_axis=-1) # (B*OL, H)
        output = tf.reshape(layer_output, [batch_size, obj_length, head_size]) # (B, OL, H)

    return output, dist



def global_workspace(inputs, gws_size, n_head, head_size, inter_size, inter_activ, atten_type, drop_rate, self_atten):

    input_shape = get_shape(inputs, expected_rank=4) # (B, M, S, H)

    batch_size = input_shape[0] # B
    n_modality = input_shape[1] # M
    seq_length = input_shape[2] # S
    feat_units = input_shape[3] # H

    inputs = tf.transpose(inputs, [2, 0, 1, 3]) # (S, B, M, H)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=seq_length) # S, (B, M, H)
    inputs_ta = inputs_ta.unstack(inputs)

    cell = tf.nn.rnn_cell.LSTMCell(gws_size)

    def loop_fn(time, cell_output, cell_state, loop_state):
        finished = (time >= seq_length)
        emit_output = cell_output
        if cell_output is None: # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            next_input = tf.zeros([batch_size, n_modality*head_size]) # (B, M*H)
            next_loop_state = tf.TensorArray(dtype=tf.float32, size=seq_length) # S, (B, N, [1, M], M)
            return finished, next_input, next_cell_state, emit_output, next_loop_state
        next_cell_state = cell_state
        obj_tensors = inputs_ta.read(time-1)
        if self_atten:
            sbj_tensors = obj_tensors # (B, M, H)
        else:
            sbj_tensors = tf.stop_gradient(cell_output[:, None, :]) # (B, 1, G)
        atten_res, atten_dist = attention(
            sbj=sbj_tensors,
            obj=obj_tensors,
            n_head=n_head, # N
            head_size=head_size, # H
            inter_size=inter_size,
            inter_activ=inter_activ,
            atten_type=atten_type,
            drop_rate=drop_rate
        ) # (B, M, H), (B, N, [1, M], M)
        next_input = tf.reshape(atten_res, [batch_size, n_modality*head_size]) # (B, M*H)
        next_loop_state = loop_state.write(time-1, atten_dist)
        return finished, next_input, next_cell_state, emit_output, next_loop_state

    outputs_ta, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
    outputs = outputs_ta.stack() # (S, B, G)
    outputs = tf.transpose(outputs, [1, 0, 2]) # (B, S, G)
    dists = loop_state_ta.stack() # (S, B, N, [1, M], M)
    dists = tf.transpose(dists, [1, 0, 2, 3, 4]) # (B, S, N, [1, M], M)

    return outputs, dists



def get_activ_fn(activ_str):
    if not isinstance(activ_str, six.string_types):
        return activ_str
    if not activ_str:
        return None
    activ = activ_str.lower()
    if activ == 'gelu':
        return lambda x: 0.5*x*(1.0+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
    else:
        return tf.keras.layers.Activation(activ)



def get_shape(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for idx, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(idx)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(tensor)
    for idx in non_static_indexes:
        shape[idx] = dyn_shape[idx]
    return shape



def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True
    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank))
        )
