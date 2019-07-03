# classifier.py
# author: Cong Bao

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import json
import random
import collections

import emoji
import numpy as np
import tensorflow as tf

import model



class Flag(object):

    def __init__(self):
        self.data_path = None
        self.ckpt_path = None
        self.output_dir = 'output/'
        self.config_file = None
        self.do_train = True
        self.do_valid = True
        self.do_test = True
        self.cv_index = 0
        self.learning_rate = 1e-3
        self.n_train_epoch = 25
        self.train_batch_size = 32
        self.valid_batch_size = 8
        self.test_batch_size = 8
        self.save_summary_steps = 5
        self.save_checkpoints_steps = 5
        self.keep_checkpoint_max = None
        self.log_step_count_steps = 1



class EmoPainExample(object):

    def __init__(self, uid, pose, emg, label):
        self.uid = uid
        self.pose = pose
        self.emg = emg
        self.label = label



class EmoPainProcessor(object):

    def __init__(self, data_dir, out_id=0):
        data = json.load(open(data_dir, 'r'))
        self.out_id = out_id
        self.pid_list = []
        self.example_dict = {}
        self.max_length = 0
        for pid, motion_dict in data.items():
            self.pid_list.append(pid)
            examples = []
            for mid, content in motion_dict.items():
                examples.append(EmoPainExample(
                    uid='{0}_{1}'.format(pid, mid),
                    pose=np.asarray(content['pose'], dtype=np.float32),
                    emg=np.asarray(content['emg'], dtype=np.float32),
                    label=self.label_class(content['pain'])
                ))
                pose_len = len(content['pose'])
                emg_len = len(content['emg'])
                assert pose_len == emg_len
                if pose_len >= self.max_length:
                    self.max_length = pose_len
            self.example_dict[pid] = examples
        self.pid_list.sort()
        assert self.out_id in range(len(self.pid_list))
        for pid in self.pid_list:
            examples = self.example_dict[pid]
            for _ in range(len(examples)):
                example = examples.pop()
                pose_shape = example.pose.shape
                emg_shape = example.emg.shape
                assert pose_shape[0] == emg_shape[0]
                diff = self.max_length - pose_shape[0]
                pose_patch = np.zeros((diff, pose_shape[1]))
                emg_patch = np.zeros((diff, emg_shape[1]))
                example.pose = np.concatenate([pose_patch, example.pose])
                example.emg = np.concatenate([emg_patch, example.emg])
                examples.insert(0, example)
            self.example_dict[pid] = examples

    @staticmethod
    def label_class(num):
        if num == 0.0:
            return 0
        else:
            return 1

    @staticmethod
    def get_n_label():
        return 2

    def get_id_list(self):
        return self.pid_list

    def get_max_seq_length(self):
        return self.max_length

    def get_train_examples(self):
        examples = []
        pid_list = self.pid_list.copy()
        pid_list.pop(self.out_id)
        for pid in pid_list:
            examples.extend(self.example_dict[pid])
        return examples

    def get_valid_examples(self):
        return self.example_dict[self.pid_list[self.out_id]]

    def get_test_examples(self):
        return self.get_valid_examples()



def input_fn_builder(examples, seq_length, batch_size, is_training):

    pose_feat_size = examples[0].pose.shape[1]
    emg_feat_size = examples[0].emg.shape[1]

    pose_examples = []
    emg_examples = []
    labels = []

    for example in examples:
        pose_examples.append(example.pose)
        emg_examples.append(example.emg)
        labels.append(example.label)
    
    def input_fn(params):
        n_example = len(examples)
        d = tf.data.Dataset.from_tensor_slices({
            'pose': tf.constant(
                value=np.asarray(pose_examples, dtype=np.float32),
                shape=[n_example, seq_length, pose_feat_size],
                dtype=tf.float32
            ),
            'emg': tf.constant(
                value=np.asarray(emg_examples, dtype=np.float32),
                shape=[n_example, seq_length, emg_feat_size],
                dtype=tf.float32
            ),
            'labels': tf.constant(
                value=labels,
                shape=[n_example],
                dtype=tf.int32
            )
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(100)

        d = d.batch(batch_size)

        return d
    
    return input_fn

def create_model(inputs, labels, config, is_training, n_label):
    
    gwt_model = model.GWTModel(
        inputs=inputs,
        config=config,
        is_training=is_training
    )

    output = gwt_model.get_projection() # (B, P)
    dists = gwt_model.get_dist_sequence() # (S, B, N, [1, M], M)

    with tf.variable_scope('loss'):
        if is_training:
            output = tf.nn.dropout(output, rate=0.1)
        logits = tf.keras.layers.Dense(n_label)(output) # (B, L)
        log_prob = tf.nn.log_softmax(logits, axis=-1) # (B, L)

        one_hot_labels = tf.one_hot(labels, depth=n_label, dtype=tf.float32) # (B, L)

        per_sample_loss = -tf.reduce_sum(one_hot_labels*log_prob, axis=-1) # (B,)
        loss = tf.reduce_mean(per_sample_loss)

    return loss, per_sample_loss, log_prob, dists

def model_fn_builder(config, n_label, learning_rate, n_train_step, init_ckpt=None):

    def model_fn(features, labels, mode, params):
        pose = features['pose']
        emg = features['emg']
        label = features['labels']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss, per_sample_loss, log_prob, dists = create_model(
            inputs=[pose, emg],
            labels=label,
            config=config,
            is_training=is_training,
            n_label=n_label
        )

        if init_ckpt is not None:
            tf.train.init_from_checkpoint(init_ckpt, {'gwt_model/': 'gwt_model/'})

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info('***************************')
            tf.logging.info('*** Trainable Variables ***')
            tf.logging.info('***************************')
            for var in tf.trainable_variables():
                tf.logging.info('  name = {0}, shape= {1}'.format(var.name, var.shape))
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=tf.train.get_global_step())
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            pred = tf.argmax(log_prob, axis=-1, output_type=tf.int32)
            acc = tf.metrics.accuracy(labels=label, predictions=pred)
            los = tf.metrics.mean(values=per_sample_loss)
            eval_metric_ops = {'eval_acc': acc, 'eval_loss': los}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )
        else:
            pred = tf.argmax(log_prob, axis=-1, output_type=tf.int32)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'pred': pred, 'dists': dists}
            )

        return output_spec

    return model_fn

def main(FLAG):

    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAG.config_file is not None:
        config = model.GWTConfig.from_json_file(FLAG.config_file)
    else:
        config = model.GWTConfig()

    tf.gfile.MakeDirs(FLAG.output_dir)

    epp = EmoPainProcessor(FLAG.data_path. FLAG.cv_index)

    n_train_step = None
    if FLAG.do_train:
        train_examples = epp.get_train_examples()
        n_train_step = int(len(train_examples)/FLAG.train_batch_size*FLAG.n_train_epoch)
        train_input_fn = input_fn_builder(
            examples=train_examples,
            seq_length=epp.get_max_seq_length(),
            batch_size=FLAG.train_batch_size,
            is_training=True
        )

    if FLAG.do_valid:
        valid_examples = epp.get_valid_examples()
        valid_input_fn = input_fn_builder(
            examples=valid_examples,
            seq_length=epp.get_max_seq_length(),
            batch_size=FLAG.valid_batch_size,
            is_training=False
        )

    if FLAG.do_test:
        test_examples = epp.get_test_examples()
        test_input_fn = input_fn_builder(
            examples=test_examples,
            seq_length=epp.get_max_seq_length(),
            batch_size=FLAG.test_batch_size,
            is_training=False
        )

    model_fn = model_fn_builder(
        config=config,
        n_label=epp.get_n_label(),
        learning_rate=FLAG.learning_rate,
        n_train_step=n_train_step,
        init_ckpt=FLAG.ckpt_path
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAG.output_dir,
        save_summary_steps=FLAG.save_summary_steps,
        save_checkpoints_steps=FLAG.save_checkpoints_steps,
        keep_checkpoint_max=FLAG.keep_checkpoint_max,
        log_step_count_steps=FLAG.log_step_count_steps
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    if FLAG.do_train and FLAG.do_valid:
        tf.logging.info('*******************************************')
        tf.logging.info('***** Running Training and Validation *****')
        tf.logging.info('*******************************************')
        tf.logging.info('  Train num examples = {}'.format(len(train_examples)))
        tf.logging.info('  Eval num examples = {}'.format(len(valid_examples)))
        tf.logging.info('  Train batch size = {}'.format(FLAG.train_batch_size))
        tf.logging.info('  Eval batch size = {}'.format(FLAG.valid_batch_size))
        tf.logging.info('  Num steps = {}'.format(n_train_step))
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=n_train_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, start_delay_secs=0, throttle_secs=0)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAG.do_train and not FLAG.do_valid:
        tf.logging.info('****************************')
        tf.logging.info('***** Running Training *****')
        tf.logging.info('****************************')
        tf.logging.info('  Num examples = {}'.format(len(train_examples)))
        tf.logging.info('  Batch size = {}'.format(FLAG.train_batch_size))
        tf.logging.info('  Num steps = {}'.format(n_train_step))
        estimator.train(input_fn=train_input_fn, max_steps=n_train_step)

    if FLAG.do_valid and not FLAG.do_train:
        tf.logging.info('******************************')
        tf.logging.info('***** Running Validation *****')
        tf.logging.info('******************************')
        tf.logging.info('  Num examples = {}'.format(len(valid_examples)))
        tf.logging.info('  Batch size = {}'.format(FLAG.valid_batch_size))
        result = estimator.evaluate(input_fn=valid_input_fn, checkpoint_path=FLAG.ckpt_path)
        output_valid_file = os.path.join(FLAG.output_dir, 'valid_results.txt')
        with tf.gfile.GFile(output_valid_file, 'w') as writer:
            tf.logging.info('***** Validation Results *****')
            for key in sorted(result.keys()):
                tf.logging.info('  {0} = {1}'.format(key, str(result[key])))
                writer.write('{0} = {1}\n'.format(key, str(result[key])))

    if FLAG.do_test:
        tf.logging.info('***************************')
        tf.logging.info('***** Running Testing *****')
        tf.logging.info('***************************')
        tf.logging.info('  Num examples = {}'.format(len(test_examples)))
        tf.logging.info('  Batch size = {}'.format(FLAG.test_batch_size))
        result = estimator.predict(input_fn=test_input_fn, checkpoint_path=FLAG.ckpt_path)
        output_test_file = os.path.join(FLAG.output_dir, 'test_results.json')
        with tf.gfile.GFile(output_test_file, 'w') as writer:
            tf.logging.info('***** Test Results *****')
            output = {}
            for i, pred in enumerate(result):
                uid = test_examples[i].uid
                lab = int(test_examples[i].label)
                res = int(pred['pred'])
                dis = pred['dists'].tolist()
                emo = emoji.emojize(':heavy_check_mark:' if lab == res else ':heavy_multiplication_x:')
                tf.logging.info('  ID: {0}\tLabel: {1}\tPrediction: {2}\t{3}'.format(uid, lab, res, emo))
                output[uid] = {'label': lab, 'pred': res, 'dist': dis}
            writer.write(json.dumps(output))

if __name__ == "__main__":
    main(Flag())
