# optimizer.py
# author: Cong Bao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf



class AdamWeightDecayOptimizer(tf.train.Optimizer):

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name='AdamWeightDecayOptimizer'):
        super(AdamWeightDecayOptimizer, self).__init__(False, name)
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for grad, param in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = self._get_variable_name(param.name)
            m = tf.get_variable(
                name=param_name + '/adam_m',
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()
            )
            v = tf.get_variable(
                name=param_name + '/adam_v',
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()
            )
            next_m = tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad)
            next_v = tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param
            update_with_lr = self.learning_rate * update
            next_param = param - update_with_lr
            assignments.extend([
                param.assign(next_param),
                m.assign(next_m),
                v.assign(next_v)
            ])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name



def create_optimizer(loss, init_lr, n_train_step, n_warmup_step=0):
    global_step = tf.train.get_or_create_global_step()
    lr = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    lr = tf.train.polynomial_decay(
        learning_rate=lr,
        global_step=global_step,
        decay_steps=n_train_step,
        end_learning_rate=5e-5
    )
    if n_warmup_step:
        global_step_int = tf.cast(global_step, tf.int32)
        warmup_step_int = tf.constant(n_warmup_step, dtype=tf.int32)
        global_step_float = tf.cast(global_step_int, tf.float32)
        warmup_step_float = tf.cast(warmup_step_int, tf.float32)
        warmup_percent_done = global_step_float / warmup_step_float
        warmup_lr = init_lr * warmup_percent_done
        is_warmup = tf.cast(global_step_int < warmup_step_int, tf.float32)
        lr = (1.0 - is_warmup) * lr + is_warmup * warmup_lr
    optimizer = AdamWeightDecayOptimizer(lr, weight_decay_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op