from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import glob
import time
import numpy as np
import pandas as pd

paths = [
    # glob.glob('/workspace/output/predictions*.simple.csv')[0],
    glob.glob('/workspace/output/predictions*.lr.csv')[0],
    glob.glob('/workspace/output/predictions*.fm.csv')[0],
    # glob.glob('/workspace/output/predictions*.tf_pairwise.csv')[0],
    # glob.glob('/workspace/output/predictions*.tf_classifier.csv')[0],
    glob.glob('/workspace/output/predictions*.gbt.csv')[0],
    glob.glob('/workspace/output/predictions*.pairwise.csv')[0]
]

def main():
    t_id = []
    probs = []
    for path in paths:
        df = pd.read_csv(path)
        t_id = df['id'].values
        probs.append(df['probability'].values)

    probability = np.power(np.prod(probs, axis=0), 1.0 / len(paths))
    assert(len(probability) == len(t_id))

    df_pred = pd.DataFrame({
        'id': t_id,
        'probability': probability,
    })
    csv_path = '/workspace/output/predictions_ensemble_{}.csv'.format(int(time.time()))
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
