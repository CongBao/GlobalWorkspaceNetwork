# model.py
# author: Cong Bao

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import random

import keras
import numpy as np
import tensorflow as tf



class AEConfig(object):

    def __init__(self):
        self.data_dir = None
        self.output_dir = 'output/'
        self.sample_step = 6
        self.win_size = 30
        self.step_size = 1
        self.batch_size = 64
        self.shuffle = True
        self.in_dims = [78, 4]
        self.hidden_size = 1024
        self.learning_rate = 1e-3
        self.n_epoch = 500



class CommSpDataGen(keras.utils.Sequence):

    def __init__(self, config):
        self.config = config
        self.pose_list = []
        self.emg_list = []
        for f in os.listdir(config.data_dir):
            path = os.path.join(config.data_dir, f)
            data = np.load(path)[::config.sample_step]
            size = int((len(data)-config.win_size)/config.step_size)+1
            for i in range(size):
                self.pose_list.append(
                    data[config.step_size*i:config.step_size*i+config.win_size,:78]
                )
                self.emg_list.append(
                    data[config.step_size*i:config.step_size*i+config.win_size,78:]
                )
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.pose_list)/self.config.batch_size)

    def __getitem__(self, idx):
        pose = self.pose_list[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
        emg = self.emg_list[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
        X1 = np.asarray(pose, dtype=np.float32)
        X2 = np.asarray(emg, dtype=np.float32)
        Y1 = np.flip(X1, axis=1)
        Y2 = np.flip(X2, axis=2)
        return [X1, X2], [Y1, Y2]

    def on_epoch_end(self):
        if self.config.shuffle:
            seed = np.random.randint(np.iinfo(np.int64).max)
            random.Random(seed).shuffle(self.pose_list)
            random.Random(seed).shuffle(self.emg_list)



class CommSpLSTMAE(object):

    def __init__(self, config):
        self.config = config
        n_dim = len(config.in_dims)
        inputs = [
            keras.layers.Input(shape=(None, d))
            for d in config.in_dims
        ]
        encoders = [
            keras.layers.CuDNNLSTM(
                units=config.hidden_size,
                return_state=True,
                name='encoder_{}'.format(i)
            )
            for i in range(n_dim)
        ]
        decoders = [
            keras.layers.CuDNNLSTM(
                units=config.hidden_size,
                return_sequences=True,
                go_backwards=True,
                name='decoder_{}'.format(i)
            )
            for i in range(n_dim)
        ]
        en_outs, en_hs, en_cs = list(zip(*[
            encoders[i](inputs[i])
            for i in range(n_dim)
        ]))
        pad_en_outs = [
            keras.layers.Lambda(lambda x: x[:,None,:])(eo)
            for eo in en_outs
        ]
        de_ins = [
            keras.layers.Lambda(lambda x: x[:,1:,:])(ipt)
            for ipt in inputs
        ]
        comm_en_h = keras.layers.add(list(en_hs))
        de_outs = [
            decoders[i](de_ins[i], initial_state=[comm_en_h, en_cs[i]])
            for i in range(n_dim)
        ]
        outs = [
            keras.layers.concatenate(
                inputs=[pad_en_outs[i], de_outs[i]],
                axis=1
            )
            for i in range(n_dim)
        ]
        out_linears = [
            keras.layers.TimeDistributed(keras.layers.Dense(d))
            for d in config.in_dims
        ]
        outputs = [
            out_linears[i](o)
            for i, o in enumerate(outs)
        ]
        self.model = keras.models.Model(inputs, outputs)
        self.cs_models = [
            keras.models.Model(inputs[i], en_outs[i])
            for i in range(n_dim)
        ]

    def train(self):
        tf.gfile.MakeDirs(self.config.output_dir)
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=self.config.learning_rate),
            loss=keras.losses.mean_squared_error,
            loss_weights=[1, 1e+6]
        )
        self.model.fit_generator(
            generator=CommSpDataGen(self.config),
            epochs=self.config.n_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.config.output_dir, 'ckpt_best.h5'),
                    monitor='loss',
                    save_best_only=True
                )
            ]
        )

    def predict(self, seq):
        res1, res2 = self.model.predict(seq, batch_size=self.config.batch_size)
        return res1[:,::-1,:], res2[:,::-1,:]

    def extract(self, pose_seq=None, emg_seq=None):
        if pose_seq is not None:
            return np.squeeze(self.cs_models[0].predict(pose_seq, batch_size=self.config.batch_size))
        if emg_seq is not None:
            return np.squeeze(self.cs_models[1].predict(emg_seq, batch_size=self.config.batch_size))



class VAEConfig(object):

    def __init__(self):
        self.data_dir = None
        self.output_dir = 'output/'
        self.sample_step = 6
        self.win_size = 30
        self.step_size = 1
        self.batch_size = 64
        self.pose_encoder = None
        self.emg_encoder = None
        self.shuffle = True
        self.in_dim = 1024
        self.latent_dim = 256
        self.hidden_size = 512
        self.out_dim = 78
        self.learning_rate = 1e-3
        self.n_epoch = 500



class VAEDataGen(keras.utils.Sequence):

    def __init__(self, config):
        self.config = config
        data_list = []
        self.label_list = []
        for f in os.listdir(config.data_dir):
            path = os.path.join(config.data_dir, f)
            data = np.load(path)[::config.sample_step]
            size = int((len(data)-config.win_size)/config.step_size)+1
            for i in range(size):
                data_list.append(
                    data[config.step_size*i:config.step_size*i+config.win_size]
                )
                self.label_list.append(
                    data[config.step_size*i+config.win_size]
                )
        self.feat_list = []
        while len(data_list) >= config.batch_size:
            batch = data_list[:config.batch_size]
            batch = np.asarray(batch, dtype=np.float32)
            if config.pose_encoder is not None:
                feat = config.pose_encoder.extract(pose_seq=batch)
            elif config.emg_encoder is not None:
                feat = config.emg_encoder.extract(emg_seq=batch)
            feat = np.squeeze(feat)
            self.feat_list.extend(feat.tolist())
            del data_list[:config.batch_size]
        if len(data_list) > 0:
            batch = np.asarray(data_list, dtype=np.float32)
            if config.pose_encoder is not None:
                feat = config.pose_encoder.extract(pose_seq=batch)
            elif config.emg_encoder is not None:
                feat = config.emg_encoder.extract(emg_seq=batch)
            feat = np.squeeze(feat)
            self.feat_list.extend(feat.tolist())

    def __len__(self):
        return int(len(self.label_list)/self.config.batch_size)

    def __getitem__(self, idx):
        X = self.feat_list[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
        X = np.asarray(X, dtype=np.float32)
        Y = self.label_list[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
        Y = np.asarray(Y, dtype=np.float32)
        return [X, Y], None

    def on_epoch_end(self):
        if self.config.shuffle:
            seed = np.random.randint(np.iinfo(np.int64).max)
            random.Random(seed).shuffle(self.feat_list)
            random.Random(seed).shuffle(self.label_list)



class FutureVAE(object):

    def __init__(self, config):
        self.config = config

        ipt = keras.layers.Input(shape=(config.in_dim,))
        z_m = keras.layers.Dense(config.latent_dim, name='z_mean')(ipt)
        z_v = keras.layers.Dense(config.latent_dim, name='z_log_var')(ipt)
        z = keras.layers.Lambda(self.sampling, name='z')([z_m, z_v])
        encoder = keras.models.Model(ipt, [z_m, z_v, z])

        latent_ipt = keras.layers.Input(shape=(config.latent_dim,))
        x = keras.layers.Dense(config.hidden_size, 'relu')(latent_ipt)
        out = keras.layers.Dense(config.out_dim)(x)
        decoder = keras.models.Model(latent_ipt, out)

        output = decoder(encoder(ipt)[2])

        label = keras.layers.Input(shape=(config.out_dim,))

        self.vae = keras.models.Model([ipt, label], output)

        recon_loss = keras.losses.mse(label, output)
        recon_loss *= config.in_dim
        kl_loss = 1 + z_v - keras.backend.square(z_m) - keras.backend.exp(z_v)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(recon_loss + kl_loss)
        self.vae.add_loss(vae_loss)

    @staticmethod
    def sampling(args):
        z_m, z_v = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        ep = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(0.5*z_v)*ep

    def train(self):
        tf.gfile.MakeDirs(self.config.output_dir)
        self.vae.compile(
            optimizer=keras.optimizers.Adam(lr=self.config.learning_rate)
        )
        self.vae.fit_generator(
            generator=VAEDataGen(self.config),
            epochs=self.config.n_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.config.output_dir, 'ckpt_best.h5'),
                    monitor='loss',
                    save_best_only=True
                )
            ]
        )

    def predict(self, x):
        return self.vae.predict(x, batch_size=self.config.batch_size)
