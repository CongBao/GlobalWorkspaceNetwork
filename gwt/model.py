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
                 map_size=128,
                 proj_size=512,
                 head_size=128,
                 gws_size=512,
                 n_head=4,
                 atten_type='general',
                 map_activ='relu',
                 proj_activ='relu',
                 value_activ='relu',
                 map_dropout=0.1,
                 proj_dropout=0.1):
        self.map_size=map_size
        self.proj_size=proj_size
        self.head_size=head_size
        self.gws_size=gws_size
        self.n_head=n_head
        self.atten_type=atten_type
        self.map_activ=map_activ
        self.proj_activ=proj_activ
        self.value_activ=value_activ
        self.map_dropout=map_dropout
        self.proj_dropout=proj_dropout
  
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
                ) # M_[(B, S, F')] -> (B, M, S, F)
            with tf.variable_scope('gws'): # dense output -> rnn cell (loop)
                self.outputs, self.dists = global_workspace(
                    inputs=self.mapped_inputs,
                    gws_size=config.gws_size,
                    n_head=config.n_head,
                    head_size=config.head_size,
                    atten_type=config.atten_type,
                    value_activ=config.value_activ
                ) # (B, M, S, F) -> (S, B, G), (S, B, N, M)
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



# share_variables = lambda func: tf.make_template(
#     name_=func.__name__,
#     func_=func,
#     create_scope_now_=True
# )

# scalar dimensions reference
# B = batch size
# M = number of modalities
# S = sequence length
# F'= feature units before mapping 
# F = feature units after mapping
# G = global workspace size
# N = number of heads
# H = size of each head (hidden size)
# P = projection size

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

#@share_variables
def general(sbj, obj, n_head, head_size):
    """ General (multiplicative) attention. (Luong 2015)
    """
    
    sbj_shape = get_shape(sbj, expected_rank=2) # (B, G)
    obj_shape = get_shape(obj, expected_rank=3) # (B, M, F)

    gws_units = sbj_shape[1] # G
    batch_size = obj_shape[0] # B
    n_modality = obj_shape[1] # M
    feat_units = obj_shape[2] # F

    obj_tensors = tf.reshape(obj, [-1, feat_units]) # (B*M, F)

    key = tf.keras.layers.Dense(
        units=n_head*gws_units,
        name='key'
    )(obj_tensors) # (B*M, N*H)

    query = tf.reshape(sbj, [batch_size, 1, 1, gws_units]) # (B, 1, 1, G)
    query = tf.tile(query, [1, n_head, 1, 1]) # (B, N, 1, G)
    key = tf.reshape(key, [batch_size, n_modality, n_head, gws_units]) # (B, M, N, G)
    key = tf.transpose(key, [0, 2, 1, 3]) # (B, N, M, G)
    scores = tf.matmul(query, key, transpose_b=True) # (B, N, 1, M)
    dist = tf.nn.softmax(scores) # (B, N, 1, M)

    return dist

#@share_variables
def additive(sbj, obj, n_head, head_size):
    """ Additive attention. (Bahdanau 2015)
    """

    sbj_shape = get_shape(sbj, expected_rank=2) # (B, G)
    obj_shape = get_shape(obj, expected_rank=3) # (B, M, F)

    gws_units = sbj_shape[1] # G
    batch_size = obj_shape[0] # B
    n_modality = obj_shape[1] # M
    feat_units = obj_shape[2] # F

    obj_tensors = tf.reshape(obj, [-1, feat_units]) # (B*M, F)

    query = tf.keras.layers.Dense(
        units=n_head*head_size,
        name='query'
    )(sbj) # (B, N*H)
    key = tf.keras.layers.Dense(
        units=n_head*head_size,
        name='key'
    )(obj_tensors) # (B*M, N*H)

    query = tf.reshape(query, [batch_size, 1, n_head, head_size]) # (B, 1, N, H)
    key = tf.reshape(key, [batch_size, n_modality, n_head, head_size]) # (B, M, N, H)
    add = tf.add(query, key) # (B, M, N, H)
    add = tf.nn.tanh(add) # (B, M, N, H)
    add = tf.reshape(add, [-1, head_size]) # (B*M*N, H)
    scores = tf.layers.dense(add, 1, name='score')
    scores = tf.reshape(scores, [batch_size, n_modality, 1, n_head]) # (B, M, 1, N)
    scores = tf.transpose(scores, [0, 3, 2, 1]) # (B, N, 1, M)
    dist = tf.nn.softmax(scores) # (B, N, 1, M)

    return dist

#@share_variables
def scaled_dot_product(sbj, obj, n_head, head_size):
    """ Scaled dot product attention. (Vaswani 2017)
    """

    sbj_shape = get_shape(sbj, expected_rank=2) # (B, G)
    obj_shape = get_shape(obj, expected_rank=3) # (B, M, F)

    gws_units = sbj_shape[1] # G
    batch_size = obj_shape[0] # B
    n_modality = obj_shape[1] # M
    feat_units = obj_shape[2] # F

    obj_tensors = tf.reshape(obj, [-1, feat_units]) # (B*M, F)

    query = tf.keras.layers.Dense(
        units=n_head*head_size,
        name='query'
    )(sbj) # (B, N*H)
    key = tf.keras.layers.Dense(
        units=n_head*head_size,
        name='key'
    )(obj_tensors) # (B*M, N*H)

    query = tf.reshape(query, [batch_size, 1, n_head, head_size]) # (B, 1, N, H)
    query = tf.transpose(query, [0, 2, 1, 3]) # (B, N, 1, H)
    key = tf.reshape(key, [batch_size, n_modality, n_head, head_size]) # (B, M, N, H)
    key = tf.transpose(key, [0, 2, 1, 3]) # (B, N, M, H)
    scores = tf.matmul(query, key, transpose_b=True) # (B, N, 1, M)
    scores = tf.multiply(scores, 1./math.sqrt(float(head_size))) # (B, N, 1, M)
    dist = tf.nn.softmax(scores) # (B, N, 1, M)

    return dist

#@share_variables
def attention(sbj, obj, n_head, head_size, score_func, value_activ=None):

    sbj_shape = get_shape(sbj, expected_rank=2) # (B, G)
    obj_shape = get_shape(obj, expected_rank=3) # (B, M, F)

    gws_units = sbj_shape[1] # G
    batch_size = obj_shape[0] # B
    n_modality = obj_shape[1] # M
    feat_units = obj_shape[2] # F

    with tf.variable_scope('attention'):
        with tf.variable_scope(score_func.__name__):
            dist = score_func(sbj, obj, n_head, head_size) # (B, N, 1, M)
            dist_r3 = tf.squeeze(dist) # (B, N, M)
        obj_tensors = tf.reshape(obj, [-1, feat_units]) # (B*M, F)
        value = tf.keras.layers.Dense(
            units=n_head*head_size,
            activation=value_activ,
            name='value'
        )(obj_tensors) # (B*M, N*H)
        value = tf.reshape(value, [batch_size, n_modality, n_head, head_size]) # (B, M, N, H)
        value = tf.transpose(value, [0, 2, 1, 3]) # (B, N, M, H)
        context = tf.matmul(dist, value) # (B, N, 1, H)
        context = tf.reshape(context, [batch_size, n_head*head_size]) # (B, N*H)

    return context, dist_r3

def global_workspace(inputs, gws_size, n_head, head_size, atten_type='general', value_activ=None):

    input_shape = get_shape(inputs, expected_rank=4) # (B, M, S, F)

    batch_size = input_shape[0] # B
    n_modality = input_shape[1] # M
    seq_length = input_shape[2] # S
    feat_units = input_shape[3] # F

    inputs = tf.transpose(inputs, [2, 0, 1, 3]) # (S, B, M, F)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=seq_length) # S, (B, M, F)
    inputs_ta = inputs_ta.unstack(inputs)

    atten_dist_ta = tf.TensorArray(dtype=tf.float32, size=seq_length) # S, (B, N, M)

    cell = tf.nn.rnn_cell.LSTMCell(gws_size)

    atten_dict = {
        'general': general,
        'additive': additive,
        'scaled_dot_product': scaled_dot_product
    }
    if atten_type not in atten_dict:
        raise ValueError('Unknown attention type: %s' % atten_type)

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None: # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            next_input, atten_dist = attention(
                sbj=tf.zeros([batch_size, gws_size]),
                obj=inputs_ta.read(time), # (B, M, F)
                n_head=n_head, # N
                head_size=head_size, # H
                score_func=atten_dict[atten_type],
                value_activ=value_activ
            )
            next_loop_state = atten_dist_ta.write(time, atten_dist)
        else: # time >= 1
            next_cell_state = cell_state
            next_input, atten_dist = attention(
                sbj=cell_output, # (B, G) ?
                obj=inputs_ta.read(time), # (B, M, F)
                n_head=n_head, # N
                head_size=head_size, # H
                score_func=atten_dict[atten_type],
                value_activ=value_activ
            )
            next_loop_state = loop_state.write(time, atten_dist)
        finished = (time >= seq_length-1)
        return finished, next_input, next_cell_state, emit_output, next_loop_state

    outputs_ta, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
    outputs = outputs_ta.stack() # (S, B, G)
    dists = loop_state_ta.stack() # (S, B, N, M)

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
