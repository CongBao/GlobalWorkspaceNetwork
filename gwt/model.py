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
  
    def __init__(self):
        pass
  
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
  
    def __init__(self, config):
        config = copy.deepcopy(config)
    
    with tf.variable_scope('gwt_model'):
        with tf.variable_scope('mapping'): # input -> dense (sequence)
            pass
        with tf.variable_scope('gws'): # dense output -> rnn cell (loop)
            pass
        with tf.variable_scope('projection'): # rnn last cell state -> fc
            pass



share_variables = lambda func: tf.make_template(
    name_=func.__name__,
    func_=func,
    create_scope_now_=True
)

# scalar dimensions reference
# B = batch size
# M = number of modalities
# S = sequence length
# F = feature units
# G = global workspace size
# N = number of heads
# H = size of each head

@share_variables
def general_attention(sbj, obj, n_head, head_size):
    pass

@share_variables
def additive_attention(sbj, obj, n_head, head_size):
    pass

@share_variables
def scaled_dot_product_attention(sbj, obj, n_head, head_size):

    sbj_shape = get_shape(sbj, expected_rank=2) # (B, G)
    obj_shape = get_shape(obj, expected_rank=3) # (B, M, F)

    gws_units = sbj_shape[1] # G
    batch_size = obj_shape[0] # B
    n_modality = obj_shape[1] # M
    feat_units = obj_shape[2] # F

    obj_tensors = tf.reshape(obj, [-1, feat_units]) # (B*M, F)

    query = tf.layers.dense(sbj, n_head*head_size, name='query') # (B, N*H)
    key = tf.layers.dense(obj_tensors, n_head*head_size, name='key') # (B*M, N*H)
    val = tf.layers.dense(obj_tensors, n_head*head_size, name='value') # (B*M, N*H)

    query = tf.reshape(query, [batch_size, 1, n_head, head_size])
    query = tf.transpose(query, [0, 2, 1, 3]) # (B, N, 1, H)
    key = tf.reshape(key, [batch_size, n_modality, n_head, head_size])
    key = tf.transpose(key, [0, 2, 1, 3]) # (B, N, M, H)
    scores = tf.matmul(query, key, transpose_b=True) # (B, N, 1, M)
    scores = tf.multiply(scores, 1./math.sqrt(float(head_size))) # (B, N, 1, M)
    scores = tf.squeeze(scores) # (B, N, M)
    dist = tf.nn.softmax(scores)
    val = tf.reshape(val, [batch_size, n_modality, n_head, head_size])
    val = tf.transpose(val, [0, 2, 1, 3]) # (B, N, M, H)
    context = tf.matmul(dist, val)
    context = tf.transpose(context, [0, 2, 1, 3]) # (B, M, N, H)
    return context, dist

def global_workspace(inputs, # (B, M, S, F)
                     gws_size=512, # G
                     n_head=1, # N
                     head_size=128, # H
                     atten_type='general'):

    input_shape = get_shape(inputs, expected_rank=4)

    batch_size = input_shape[0] # B
    n_modality = input_shape[1] # M
    seq_length = input_shape[2] # S
    # feat_units = input_shape[3] # F

    inputs = tf.transpose(inputs, [2, 0, 1, 3]) # (S, B, M, F)
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=seq_length) # S, (B, M, F)
    inputs_ta = inputs_ta.unstack(inputs)

    atten_dist_ta = tf.TensorArray(dtype=tf.float32, size=seq_length) # S, (B, N, M)

    cell = tf.nn.rnn_cell.LSTMCell(gws_size)

    atten_dict = {
        'general': general_attention,
        'additive': additive_attention,
        'scaled_dot_product': scaled_dot_product_attention
    }
    if atten_type not in atten_dict:
        raise ValueError('Unknown attention type: %s' % atten_type)

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None: # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            next_input = inputs_ta.read(time) # (B, M, F)
            next_input = tf.reduce_mean(next_input, axis=1) # (B, F)
            next_input = tf.tile(next_input, [1, n_head]) # (B, N*F)
            next_loop_state = atten_dist_ta.write(time, tf.fill([batch_size, n_head, n_modality], .5))
        else: # time >= 1
            next_cell_state = cell_state
            next_input, atten_dist = atten_dict[atten_type](
                sbj=cell_output, # (B, G) ?
                obj=inputs_ta.read(time), # (B, M, F)
                n_head=n_head, # N
                head_size=head_size # H
            )
            next_loop_state = loop_state.write(time, atten_dist)
        finished = (time >= seq_length)
        return finished, next_input, next_cell_state, emit_output, next_loop_state

    outputs_ta, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn)
    outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2]) # (S, B, G) -> (B, S, G)
    dists = tf.transpose(loop_state_ta.stack(), [1, 0, 2, 3]) # (S, B, N, M) -> (B, S, N, M)

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