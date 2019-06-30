# analysis.py
# author: Cong Bao

import os
import json

import emoji
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def print_result_list(res_path):
    res = json.load(open(res_path, 'r'))
    cir = emoji.emojize(':black_medium_square:')
    for uid, item in res.items():
        res = item['pred']
        lab = item['label']
        emo = emoji.emojize(':heavy_check_mark:' if lab == res else ':heavy_multiplication_x:')
        print('{0}  ID: {1}\tLabel: {2}\tPrediction: {3}\t{4}'.format(cir, uid, lab, res, emo))

def print_dist_lines(res_path, out_path):
    res = json.load(open(res_path, 'r'))
    cir = emoji.emojize(':black_medium_square:')
    for uid, item in res.items():
        res = item['pred']
        lab = item['label']
        emo = emoji.emojize(':heavy_check_mark:' if lab == res else ':heavy_multiplication_x:')
        print('{0}  ID: {1}\tLabel: {2}\tPrediction: {3}\t{4}\n'.format(cir, uid, lab, res, emo))
        dist = np.asarray(item['dist'], dtype=np.float32)
        hs, ms = dist.shape[1], dist.shape[2]
        fig, axes = plt.subplots(hs, ms, figsize=[ms*9, hs*3.5])
        for h in range(hs):
            for m in range(ms):
                axes[h, m].plot(dist[:,h,m,:])
                axes[h, m].set_ylim(0, 1)
                axes[h, m].set_title('Head {0}, Modality {1}'.format(h, m))
                axes[h, m].legend(['M{}'.format(i) for i in range(dist.shape[-1])])
        fig.savefig(os.path.join(out_path, '{}.png'.format(uid)), dpi=fig.dpi)
    plt.close('all')

def print_dist_stack_lines(res_path, out_path):
    res = json.load(open(res_path, 'r'))
    cir = emoji.emojize(':black_medium_square:')
    for uid, item in res.items():
        res = item['pred']
        lab = item['label']
        emo = emoji.emojize(':heavy_check_mark:' if lab == res else ':heavy_multiplication_x:')
        print('{0}  ID: {1}\tLabel: {2}\tPrediction: {3}\t{4}\n'.format(cir, uid, lab, res, emo))
        dist = np.asarray(item['dist'], dtype=np.float32)
        hs, ms = dist.shape[1], dist.shape[2]
        fig, axes = plt.subplots(hs, ms, figsize=[ms*9, hs*3.5])
        for h in range(hs):
            for m in range(ms):
                x = list(range(len(dist)))
                y = [dist[:,h,m,i] for i in range(ms)]
                l = ['M{}'.format(i) for i in range(dist.shape[-1])]
                axes[h, m].stackplot(x, *y, labels=l)
                axes[h, m].set_ylim(0, 1)
                axes[h, m].set_title('Head {0}, Modality {1}'.format(h, m))
                axes[h, m].legend(['M{}'.format(i) for i in range(dist.shape[-1])])
        fig.savefig(os.path.join(out_path, '{}.png'.format(uid)), dpi=fig.dpi)
    plt.close('all')
