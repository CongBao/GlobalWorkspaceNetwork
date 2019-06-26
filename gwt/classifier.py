# classifier.py
# author: Cong Bao

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

import model

def create_model(inputs, labels, config, is_training, n_labels):
    
    gwt_model = model.GWTModel(
        inputs=inputs,
        config=config,
        is_training=is_training
    )

    output = gwt_model.get_projection() # (B, P)

    with tf.variable_scope('loss'):
        if is_training:
            output = tf.nn.dropout(output, rate=0.1)
        logits = tf.keras.layers.Dense(n_labels)(output) # (B, L)
        log_prob = tf.nn.log_softmax(logits, axis=-1) # (B, L)

        one_hot_labels = tf.one_hot(labels, depth=n_labels, dtype=tf.float32)

        per_sample_loss = -tf.reduce_sum(one_hot_labels*log_prob, axis=-1)
        loss = tf.reduce_mean(per_sample_loss)

    return loss, per_sample_loss, logits
