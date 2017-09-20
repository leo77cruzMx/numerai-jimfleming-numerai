from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd

from tpot import TPOTClassifier

import os

def main():
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

    prefix = os.getenv('PREFIX', '/workspace/output/')
    tsne_data = np.load('{}tsne_2d_5p.npz'.format(prefix))
    tsne_train = tsne_data['train']
    tsne_valid = tsne_data['valid']
    tsne_test = tsne_data['test']

    # concat features
    X_train_concat = np.concatenate([X_train, tsne_train], axis=1)
    X_valid_concat = np.concatenate([X_valid, tsne_valid], axis=1)
    X_test_concat = np.concatenate([X_test, tsne_test], axis=1)

    tpot = TPOTClassifier(
        max_time_mins=60 * 24,
        population_size=100,
        scoring='log_loss',
        cv=3,
        verbosity=2,
        random_state=67)
    tpot.fit(X_train_concat, y_train)
    loss = tpot.score(X_valid_concat, y_valid)
    print(loss)
    tpot.export('{}tpot_pipeline.py'.format(prefix))

    p_test = tpot.predict_proba(X_test_concat)
    df_pred = pd.DataFrame({
        'id': df_test['id'],
        'probability': p_test[:,1]
    })
    csv_path = os.getenv('PREDICTING', '/workspace/output/predictions_{}.tpot.csv'.format(loss))
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
