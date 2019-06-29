# model.py
# author: Cong Bao

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
import copy
import json
import math

import tensorflow as tf



class GWTConfig(object):

    def __init__(self,
                 map_size=32, # 128
                 proj_size=64, # 512
                 head_size=32, # 128
                 gws_size=64, # 512
                 n_head=2, # 4
                 self_atten=0,
                 atten_type='general',
                 map_activ=None,
                 proj_activ='relu',
                 value_activ=None,
                 map_dropout=0.1,
                 proj_dropout=0.1):
        self.map_size = map_size
        self.proj_size = proj_size
        self.head_size = head_size
        self.gws_size = gws_size
        self.n_head = n_head
        self.self_atten = self_atten
        self.atten_type = atten_type
        self.map_activ = map_activ
        self.proj_activ = proj_activ
        self.value_activ = value_activ
        self.map_dropout = map_dropout
        self.proj_dropout = proj_dropout

    @classmethod
    def from_dict(cls, json_obj):
        config = GWTConfig()
        for k, v in six.iteritems(json_obj):
            config.__dict__[k] = v
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'



class GWTModel(object):

    def __init__(self, inputs, config, is_training):

        config = copy.deepcopy(config)
        if not is_training:
            config.map_dropout = 0.0
            config.proj_dropout = 0.0

        with tf.variable_scope('gwt_model'):
            with tf.variable_scope('mapping'): # input -> dense (sequence)
                self.mapped_inputs = mapping(
                    input_list=inputs,
                    units=config.map_size,
                    activ=config.map_activ,
                    dropout=config.map_dropout
                ) # M, (B, S, F') -> (B, M, S, F)
            with tf.variable_scope('gws'): # dense output -> rnn cell (loop)
                self.outputs, self.dists = global_workspace(
                    inputs=self.mapped_inputs,
                    gws_size=config.gws_size,
                    n_head=config.n_head,
                    head_size=config.head_size,
                    atten_type=config.atten_type,
                    self_atten=config.self_atten,
                    value_activ=config.value_activ
                ) # (B, M, S, F) -> (S, B, G), (S, B, N, [1,M,1+M], M)
            with tf.variable_scope('projection'): # rnn last cell state -> fc
                self.proj_output = projection(
                    features=self.outputs[-1],
                    units=config.proj_size,
                    activ=config.proj_activ,
                    dropout=config.proj_dropout
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
            config.proj_dropout = 0.0
        
        with tf.variable_scope('conc_model'):
            self.outputs, _ = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.LSTMCell(config.gws_size),
                inputs=tf.concat(inputs, axis=-1), # M, (B, S, F') -> (B, S, F*)
                dtype=tf.float32
            )
            with tf.variable_scope('projection'):
                self.proj_output = projection(
                    features=self.outputs[:,-1,:],
                    units=config.proj_size,
                    activ=config.proj_activ,
                    dropout=config.proj_dropout
                ) # (B, G) -> (B, P)

    def get_projection(self):
        return self.proj_output

    def get_gws_sequence(self):
        return self.outputs



class MapConcModel(object):

    def __init__(self, inputs, config, is_training):

        config = copy.deepcopy(config)
        if not is_training:
            config.map_dropout = 0.0
            config.proj_dropout = 0.0
        
        with tf.variable_scope('map_conc_model'):
            with tf.variable_scope('mapping'): # input -> dense (sequence)
                self.mapped_inputs = mapping(
                    input_list=inputs,
                    units=config.map_size,
                    activ=config.map_activ,
                    dropout=config.map_dropout
                ) # M, (B, S, F') -> (B, M, S, F)
            input_list = tf.unstack(self.mapped_inputs, axis=1) # (B, M, S, F) -> M, (B, S, F)
            inputs = tf.concat(inputs, axis=-1) # M, (B, S, F) -> (B, S, 2*F)
            self.outputs, _ = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.LSTMCell(config.gws_size),
                inputs=inputs,
                dtype=tf.float32
            )
            with tf.variable_scope('projection'):
                self.proj_output = projection(
                    features=self.outputs[:,-1,:],
                    units=config.proj_size,
                    activ=config.proj_activ,
                    dropout=config.proj_dropout
                ) # (B, G) -> (B, P)

    def get_projection(self):
        return self.proj_output

    def get_gws_sequence(self):
        return self.outputs



# scalar dimensions reference
#  B = batch size
#  M = number of modalities
#  S = sequence length
#  F'= feature size before mapping 
#  F = feature size after mapping
#  G = global workspace size
#  N = number of heads
#  H = size of each head (hidden size)
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
        features = input_list[idx] # (B, S, F')
        feat_shape = get_shape(features, expected_rank=3)
        feats = tf.nn.dropout(
            x=features,
            rate=dropout,
            noise_shape=(feat_shape[0], 1, feat_shape[2])
        ) # (B, S, F')
        feats = tf.keras.layers.TimeDistributed(
            layer=tf.keras.layers.Dense(units, activation=activ),
            name='map_{}'.format(idx)
        )(feats) # (B, S, F)
        mapped.append(feats)

    inputs = tf.stack(mapped) # (M, B, S, F)
    inputs = tf.transpose(inputs, [1, 0, 2, 3]) # (B, M, S, F)
    
    return inputs

def projection(features, units, activ=None, dropout=0.0):
    
    assert_rank(features, 2) # (B, G)

    proj = tf.nn.dropout(features, rate=dropout)
    proj = tf.keras.layers.Dense(
        units=units,
        activation=activ
    )(proj) # (B, P)

    return proj

def attention(sbj, obj, n_head, head_size, atten_type, value_activ=None):

    sbj_shape = get_shape(sbj, expected_rank=3) # (B, SL, SF) : (B, 1, G) or (B, M, F)
    obj_shape = get_shape(obj, expected_rank=3) # (B, OL, OF) : (B, M, F)

    batch_size = sbj_shape[0] # B
    sbj_length = sbj_shape[1] # SL : 1 or M
    obj_length = obj_shape[1] # OL : M
    sbj_feats = sbj_shape[2] # SF : G or F
    obj_feats = obj_shape[2] # OF : F

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
        with tf.variable_scope(atten_dict[atten_type].__name__):
            dist = atten_dict[atten_type](sbj, obj, n_head, head_size) # (B, N, SL, OL)
        obj_tensors = tf.reshape(obj, [-1, obj_feats]) # (B*OL, OF)
        value = tf.keras.layers.Dense(
            units=n_head*head_size,
            activation=value_activ,
            name='value'
        )(obj_tensors) # (B*OL, N*H)
        value = tf.reshape(value, [batch_size, obj_length, n_head, head_size]) # (B, OL, N, H)
        value = tf.transpose(value, [0, 2, 1, 3]) # (B, N, OL, H)
        context = tf.matmul(dist, value) # (B, N, SL, H)
        context = tf.transpose(context, [0, 2, 1, 3]) # (B, SL, N, H)
        context = tf.reshape(context, [batch_size, sbj_length, n_head*head_size]) # (B, SL, N*H)

    return context, dist

def global_workspace(inputs, gws_size, n_head, head_size, atten_type='general', self_atten=0, value_activ=None):

    input_shape = get_shape(inputs, expected_rank=4) # (B, M, S, F)

    batch_size = input_shape[0] # B
    n_modality = input_shape[1] # M
    seq_length = input_shape[2] # S
    feat_units = input_shape[3] # F

    inputs = tf.transpose(inputs, [2, 0, 1, 3]) # (S, B, M, F)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=seq_length) # S, (B, M, F)
    inputs_ta = inputs_ta.unstack(inputs)

    atten_dist_ta = tf.TensorArray(dtype=tf.float32, size=seq_length) # S, (B, N, [1,M,1+M], M)

    cell = tf.nn.rnn_cell.LSTMCell(gws_size)

    def loop_fn(time, cell_output, cell_state, loop_state):
        finished = (time >= seq_length)
        emit_output = cell_output
        if cell_output is None: # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            if self_atten == 0:
                next_input = tf.zeros([batch_size, n_head*head_size]) # (B, N*H)
                next_loop_state = atten_dist_ta.write(time, tf.zeros([batch_size, n_head, 1, n_modality])) # (B, N, 1, M)
            if self_atten == 1:
                next_input = tf.zeros([batch_size, n_modality*n_head*head_size]) # (B, M*N*H)
                next_loop_state = atten_dist_ta.write(time, tf.zeros([batch_size, n_head, n_modality, n_modality])) # (B, N, M, M)
            if self_atten == 2:
                next_input = tf.zeros([batch_size, (1+n_modality)*n_head-head_size]) # (B, (1+M)*N*H)
                next_loop_state = atten_dist_ta.write(time, tf.zeros([batch_size, n_head, 1+n_modality, n_modality])) # (B, N, (1+M), M)
            return finished, next_input, next_cell_state, emit_output, next_loop_state
        next_cell_state = cell_state
        if self_atten == 0 or self_atten == 2: # gws attention or both
            gws_atten_res, gws_atten_dist = attention(
                sbj=cell_output[:, None, :], # (B, 1, G)
                obj=inputs_ta.read(time-1), # (B, M, F)
                n_head=n_head, # N
                head_size=head_size, # H
                atten_type=atten_type,
                value_activ=value_activ
            ) # (B, 1, N*H), (B, N, 1, M)
        if self_atten == 1 or self_atten == 2: # self attention or both
            inputs_tensor = inputs_ta.read(time-1)
            self_atten_res, self_atten_dist = attention(
                sbj=inputs_tensor, # (B, M, F)
                obj=inputs_tensor, # (B, M, F)
                n_head=n_head, # N
                head_size=head_size, # H
                atten_type=atten_type,
                value_activ=value_activ
            ) # (B, M, N*H), (B, N, M, M)
        if self_atten == 0: # gws attention
            next_input = tf.reshape(gws_atten_res, [batch_size, n_head*head_size]) # (B, N*H)
            next_loop_state = loop_state.write(time-1, gws_atten_dist)
        if self_atten == 1: # self attention
            next_input = tf.reshape(self_atten_res, [batch_size, n_modality*n_head*head_size]) # (B, M*N*H)
            next_loop_state = loop_state.write(time-1, self_atten_dist)
        if self_atten == 2: # both
            next_input = tf.concat([gws_atten_res, self_atten_res], 1) # (B, 1+M, N*H)
            next_input = tf.reshape(next_input, [batch_size, (1+n_modality)*n_head-head_size]) # (B, (1+M)*N*H)
            atten_dist = tf.concat([gws_atten_dist, self_atten_dist], 2) # (B, N, (1+M), M)
            next_loop_state = loop_state.write(time-1, atten_dist)
        return finished, next_input, next_cell_state, emit_output, next_loop_state

    outputs_ta, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
    outputs = outputs_ta.stack() # (S, B, G)
    dists = loop_state_ta.stack() # (S, B, N, [1,M,1+M], M)

    return outputs, dists

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
