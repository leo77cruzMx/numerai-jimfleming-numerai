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
import sys


def locate(prefix, perplexity, polynomial, dimensions):
    if polynomial:
        name = 'tsne_{}d_{}p_poly.npz'.format(dimensions, perplexity)
    else:
        name = 'tsne_{}d_{}p.npz'.format(dimensions, perplexity)
    return os.path.join(prefix, name)

def save_tsne(perplexity, dimensions=2, polynomial=False):
    prefix = os.getenv('STORING')
    df_train = pd.read_csv(os.getenv('PREPARED_TRAINING'))
    df_valid = pd.read_csv(os.getenv('PREPARED_VALIDATING'))
    df_test = pd.read_csv(os.getenv('PREPARED_TESTING'))

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

    sys.stdout.write('Running TSNE (perplexity: {}, dimensions: {}, polynomial: {})...\n'.format(perplexity, dimensions, polynomial))
    sys.stdout.flush()
    start_time = time.time()
    tsne_all = bh_sne(X_all, d=dimensions, perplexity=float(perplexity))
    sys.stdout.write('TSNE: {}s\n'.format(time.time() - start_time))
    sys.stdout.flush()

    tsne_train = tsne_all[:X_train.shape[0]]
    assert(len(tsne_train) == len(X_train))

    tsne_valid = tsne_all[X_train.shape[0]:X_train.shape[0]+X_valid.shape[0]]
    assert(len(tsne_valid) == len(X_valid))

    tsne_test = tsne_all[X_train.shape[0]+X_valid.shape[0]:X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]]
    assert(len(tsne_test) == len(X_test))

    if polynomial:
        save_path = 'tsne_{}d_{}p_poly.npz'.format(dimensions, perplexity)
    else:
        save_path = 'tsne_{}d_{}p.npz'.format(dimensions, perplexity)
    save_path = os.path.join(prefix, save_path)

    np.savez(save_path, \
        train=tsne_train, \
        valid=tsne_valid, \
        test=tsne_test)
    sys.stdout.write('Saved: {}\n'.format(save_path))
    sys.stdout.flush()

class Worker(threading.Thread):
    def __init__(self, tasks):
        threading.Thread.__init__(self)
        self.daemon = True
        self.tasks = tasks

    def run(self):
        prefix = os.getenv('STORING')
        while True:
            perplexity, polynomial, dimensions = self.tasks.get()
            location = locate(prefix, perplexity, polynomial, dimensions)
            delay = 450
            while True:
                sys.stdout.write('Generating {}\n'.format(location))
                sys.stdout.flush()
                os.system('python3 {} {} {} {}'.format(__file__, perplexity, 1 if polynomial else 0, dimensions))
                if os.path.isfile(location):
                    break
                else:
                    sys.stdout.write('File {} missing, backing off\n'.format(location))
                    sys.stdout.flush()
                    time.sleep(delay)
                if delay < 3600:
                    delay *= 2
            self.tasks.task_done()


def main():
    try:
        if len(sys.argv) < 4:
            definitions = [
                (5, False, 2), (10, False, 2), (15, False, 2), (30, False, 2), (50, False, 2),
                (5, True, 2), (10, True, 2), (15, True, 2), (30, True, 2), (50, True, 2)
            ]
            count = int(os.environ.get('PARALLEL', multiprocessing.cpu_count() / 4))
            if count <= 0:
                count = 1
            sys.stdout.write('Parallel count: {}\n'.format(count))
            sys.stdout.flush()
            tasks = queue.Queue(count)
            for _ in range(count):
                Worker(tasks).start()
            for definition in definitions:
                tasks.put(definition)
            tasks.join()
        else:
            perplexity = int(sys.argv[1])
            polynomial = bool(int(sys.argv[2]))
            dimensions = int(sys.argv[3])
            save_tsne(perplexity, polynomial=polynomial, dimensions=dimensions)
    except:
        traceback.print_exc()

if __name__ == '__main__':
    main()
