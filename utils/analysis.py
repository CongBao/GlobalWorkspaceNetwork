# analysis.py
# author: Cong Bao

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import itertools

import emoji
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics



class Analysis(object):
  
    def __init__(self, res_path='/content/output/test_results.json', n_label=3):
        self.n_label = n_label
        self.res = json.load(open(res_path, 'r'))
        self.B = emoji.emojize(':black_medium_square:')
        self.T = emoji.emojize(':heavy_check_mark:')
        self.F = emoji.emojize(':heavy_multiplication_x:')
    
    def title(self, uid, label, pred, other=''):
        mark = self.T if pred == label else self.F
        print('{0}  ID: {1}\tLabel: {2}\t Pred: {3}\t{4}{5}'.format(self.B, uid, label, pred, mark, other))
    
    def result_list(self):
        for uid, item in self.res.items():
            self.title(uid, item['label'], item['pred'])
      
    def confusion_matrix(self):
        label_list = []
        pred_list = []
        for item in self.res.values():
            label_list.append(item['label'])
            pred_list.append(item['pred'])
        metric_report(label_list, pred_list, self.n_label)
  
    def dist_lines(self):
        for uid, item in self.res.items():
            self.title(uid, item['label'], item['pred'], '\n')
            dist = np.asarray(item['dist'], dtype=np.float32)
            _, hs, ss, os = dist.shape
            _, axes = plt.subplots(hs, ss, figsize=[ss*9, hs*3.5])
            if hs == 1:
                if ss == 1:
                    axes.plot(dist[:,0,0,:])
                    axes.set_ylim(0, 1)
                    axes.set_title('Head 0')
                    axes.legend(['M{}'.format(i) for i in range(os)])
                    continue
                for s in range(ss):
                    axes[s].plot(dist[:,0,s,:])
                    axes[s].set_ylime(0, 1)
                    axes[s].set_title('Head 0')
                    axes[s].legend(['M{}'.format(i) for i in range(os)])
                continue
            for h in range(hs):
                if ss == 1:
                    axes[h].plot(dist[:,h,0,:])
                    axes[h].set_ylim(0, 1)
                    axes[h].set_title('Head {}'.format(h))
                    axes[h].legend(['M{}'.format(i) for i in range(os)])
                    continue
                for s in range(ss):
                    axes[h, s].plot(dist[:,h,s,:])
                    axes[h, s].set_ylim(0, 1)
                    axes[h, s].set_title('Head {0}, Modality {1}'.format(h, s))
                    axes[h, s].legend(['M{}'.format(i) for i in range(os)])
            plt.show()
            print('\n')
      
    def dist_stack_lines(self):
        for uid, item in self.res.items():
            self.title(uid, item['label'], item['pred'], '\n')
            dist = np.asarray(item['dist'], dtype=np.float32)
            sl, hs, ss, os = dist.shape
            _, axes = plt.subplots(hs, ss, figsize=[ss*9, hs*3.5])
            x = list(range(sl))
            if hs == 1:
                if ss == 1:
                    y = [dist[:,0,0,i] for i in range(os)]
                    l = ['M{}'.format(i) for i in range(os)]
                    axes.stackplot(x, *y, labels=l)
                    axes.set_ylim(0, 1)
                    axes.set_title('Head 0')
                    axes.legend(['M{}'.format(i) for i in range(os)])
                    continue
                for s in range(ss):
                    y = [dist[:,0,s,i] for i in range(os)]
                    l = ['M{}'.format(i) for i in range(os)]
                    axes[s].stackplot(x, *y, labels=l)
                    axes[s].set_ylim(0, 1)
                    axes[s].set_title('Head 0')
                    axes[s].legend(['M{}'.format(i) for i in range(os)])
                continue
            for h in range(hs):
                if ss == 1:
                    y = [dist[:,h,0,i] for i in range(os)]
                    l = ['M{}'.format(i) for i in range(os)]
                    axes[h].stackplot(x, *y, labels=l)
                    axes[h].set_ylim(0, 1)
                    axes[h].set_title('Head {0}'.format(h))
                    axes[h].legend(['M{}'.format(i) for i in range(os)])
                    continue
                for s in range(ss):
                    y = [dist[:,h,s,i] for i in range(os)]
                    l = ['M{}'.format(i) for i in range(os)]
                    axes[h, s].stackplot(x, *y, labels=l)
                    axes[h, s].set_ylim(0, 1)
                    axes[h, s].set_title('Head {0}, Modality {1}'.format(h, s))
                    axes[h, s].legend(['M{}'.format(i) for i in range(os)])
            plt.show()
            print('\n')



def loso_cv(res_dir=None, n_label=3):
    label_list = []
    pred_list = []
    for f in os.listdir(res_dir):
        if f.endswith('.json'):
            path = os.path.join(res_dir, f)
            res = json.load(open(path, 'r'))
            for item in res.values():
                label_list.append(item['label'])
                pred_list.append(item['pred'])
    metric_report(label_list, pred_list, n_label)



def ttest_losocv(res1_dir=None, res2_dir=None, n_label=3):
    is_json = lambda x: x.endswith('.json')
    files1 = list(filter(is_json, os.listdir(res1_dir)))
    files2 = list(filter(is_json, os.listdir(res2_dir)))
    assert len(files1) == len(files2)



def ttest_52cv(res1_dir=None, res2_dir=None, n_label=3, metric=metrics.accuracy_score):
    is_json = lambda x: x.endswith('.json')
    files1 = list(filter(is_json, os.listdir(res1_dir)))
    files2 = list(filter(is_json, os.listdir(res2_dir)))
    assert len(files1) == len(files2)

    diff = np.zeros((5, 2))
    for f1, f2 in zip(sorted(files1), sorted(files2)):
        path1 = os.path.join(res1_dir, f1)
        path2 = os.path.join(res2_dir, f2)
        res1 = json.load(open(path1, 'r'))
        res2 = json.load(open(path2, 'r'))
        lab1 = [it['label'] for it in res1.values()]
        lab2 = [it['label'] for it in res2.values()]
        pred1 = [it['pred'] for it in res1.values()]
        pred2 = [it['pred'] for it in res2.values()]
        score1 = metric(lab1, pred1)
        score2 = metric(lab2, pred2)
        name1 = f1.replace('.json', '')
        name2 = f2.replace('.json', '')
        _, nt1, nf1 = name1.split('_')
        _, nt2, nf2 = name2.split('_')
        assert nt1 == nt2
        assert nf1 == nf2
        diff[int(nt1), int(nf1)] = score1 - score2

    p_i_bar = np.mean(diff, axis=1)
    s_i_sqr = np.sum(np.square(diff - p_i_bar), axis=1)
    t = diff[0, 0] / np.sqrt(np.mean(s_i_sqr))
    return t



def metric_report(label_list, pred_list, n_label):
    cm = metrics.confusion_matrix(label_list, pred_list, labels=np.arange(n_label))
    accuracy = np.trace(cm) / np.sum(cm)
    mcc = metrics.matthews_corrcoef(label_list, pred_list)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    precision_str = ('[' + ', '.join(['%.4f']*len(precision)) + ']') % tuple(precision)
    precision_avg = np.mean(precision)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    recall_str = ('[' + ', '.join(['%.4f']*len(recall)) + ']') % tuple(recall)
    recall_avg = np.mean(recall)
    f1 = 2*precision*recall/(precision+recall)
    f1_str = ('[' + ', '.join(['%.4f']*len(f1)) + ']') % tuple(f1)
    f1_avg = 2*precision_avg*recall_avg/(precision_avg+recall_avg)
    print('Accuracy:  {}'.format(accuracy))
    print('MCC:       {}'.format(mcc))
    print('Precision: {0},\taverage: {1:.4f}'.format(precision_str, precision_avg))
    print('Recall:    {0},\taverage: {1:.4f}'.format(recall_str, recall_avg))
    print('F1 score:  {0},\taverage: {1:.4f}'.format(f1_str, f1_avg))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    label_names = np.arange(n_label)
    tick_marks = np.arange(n_label)
    plt.xticks(tick_marks, label_names, fontsize=15)
    plt.yticks(tick_marks, label_names, fontsize=15)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:,}'.format(cm[i, j]),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black',
                 fontsize=15)
    plt.tight_layout()
    plt.ylabel('Label', fontsize=15)
    plt.xlabel('Prediction', fontsize=15)
    plt.show()
    