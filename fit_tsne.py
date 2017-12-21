#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.preprocessing import PolynomialFeatures


def save_tsne(perplexity, dimensions, polynomial):
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

    template = 'Running TSNE; perplexity: {}, dimensions: {}, polynomial: {}\n'
    sys.stdout.write(template.format(perplexity, dimensions, polynomial))
    sys.stdout.flush()
    start_time = time.time()
    count = int(os.environ.get('PARALLEL', -1))
    tsne_all = TSNE(
        n_components=dimensions, perplexity=perplexity, n_jobs=count
    ).fit_transform(X_all)
    sys.stdout.write('TSNE: {}s\n'.format(time.time() - start_time))
    sys.stdout.flush()

    end = X_train.shape[0]
    tsne_train = tsne_all[:end]
    assert(len(tsne_train) == len(X_train))

    begin = X_train.shape[0]
    end = X_train.shape[0] + X_valid.shape[0]
    tsne_valid = tsne_all[begin:end]
    assert(len(tsne_valid) == len(X_valid))

    begin = X_train.shape[0] + X_valid.shape[0]
    end = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]
    tsne_test = tsne_all[begin:end]
    assert(len(tsne_test) == len(X_test))

    if polynomial:
        save_path = 'tsne_{}d_{}p_poly.npz'.format(dimensions, perplexity)
    else:
        save_path = 'tsne_{}d_{}p.npz'.format(dimensions, perplexity)
    save_path = os.path.join(prefix, save_path)

    np.savez(save_path, train=tsne_train, valid=tsne_valid, test=tsne_test)
    sys.stdout.write('Saved: {}\n'.format(save_path))
    sys.stdout.flush()


def main():
    definitions = [
        (5, False, 2), (10, False, 2), (15, False, 2),
        (30, False, 2), (50, False, 2),
        (5, True, 2), (10, True, 2), (15, True, 2),
        (30, True, 2), (50, True, 2)
    ]
    for definition in definitions:
        perplexity, polynomial, dimensions = definition
        save_tsne(perplexity, dimensions, polynomial)


if __name__ == '__main__':
    main()
