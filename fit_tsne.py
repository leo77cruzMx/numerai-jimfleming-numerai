from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd

from tsne import bh_sne
from sklearn.preprocessing import PolynomialFeatures

import os
import multiprocessing
import queue
import threading
import traceback

def save_tsne(perplexity, dimensions=2, polynomial=False):
    prefix = os.getenv('PREFIX', '/workspace/output/')
    df_train = pd.read_csv(os.getenv('TRAINING', '/workspace/output/train_data.csv'))
    df_valid = pd.read_csv(os.getenv('VALIDATING', '/workspace/output/valid_data.csv'))
    df_test = pd.read_csv(os.getenv('TESTING', '/workspace/output/test_data.csv'))

    feature_cols = list(df_train.columns[:-1])
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values

    X_test = df_test[feature_cols].values

    X_all = np.concatenate([X_train, X_valid, X_test], axis=0)

    if polynomial:
        poly = PolynomialFeatures(degree=2)
        X_all = poly.fit_transform(X_all)

    print('Running TSNE (perplexity: {}, dimensions: {}, polynomial: {})...'.format(perplexity, dimensions, polynomial))
    start_time = time.time()
    tsne_all = bh_sne(X_all, d=dimensions, perplexity=float(perplexity))
    print('TSNE: {}s'.format(time.time() - start_time))

    tsne_train = tsne_all[:X_train.shape[0]]
    assert(len(tsne_train) == len(X_train))

    tsne_valid = tsne_all[X_train.shape[0]:X_train.shape[0]+X_valid.shape[0]]
    assert(len(tsne_valid) == len(X_valid))

    tsne_test = tsne_all[X_train.shape[0]+X_valid.shape[0]:X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]]
    assert(len(tsne_test) == len(X_test))

    if polynomial:
        save_path = '{}tsne_{}d_{}p_poly.npz'.format(prefix, dimensions, perplexity)
    else:
        save_path = '{}tsne_{}d_{}p.npz'.format(prefix, dimensions, perplexity)

    np.savez(save_path, \
        train=tsne_train, \
        valid=tsne_valid, \
        test=tsne_test)
    print('Saved: {}'.format(save_path))

class Worker(threading.Thread):
    def __init__(self, tasks):
        threading.Thread.__init__(self)
        self.daemon = True
        self.tasks = tasks

    def run(self):
        while True:
            perplexity, polynomial, dimensions = self.tasks.get()
            try:
                save_tsne(perplexity, polynomial=polynomial, dimensions=dimensions)
            except:
                traceback.print_exc()
            finally:
                self.tasks.task_done()

def main():
    try:
        definitions = [
            (5, False, 2), (10, False, 2), (15, False, 2), (30, False, 2), (50, False, 2),
            (5, True, 2), (10, True, 2), (15, True, 2), (30, True, 2), (50, True, 2),
            (30, False, 3)
        ]
        count = multiprocessing.cpu_count()
        tasks = queue.Queue(count)
        for _ in range(count):
            Worker(tasks).start()
        for definition in definitions:
            tasks.put(definition)
        tasks.join()
    except:
        traceback.print_exc()

if __name__ == '__main__':
    main()
