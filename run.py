import glob
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import os

def merge_tsne(selection):
    prefix = os.getenv('PREFIX', '/workspace/output/')
    each = []
    each.append('{}tsne_2d_5p.npz'.format(prefix))
    each.append('{}tsne_2d_10p.npz'.format(prefix))
    each.append('{}tsne_2d_15p.npz'.format(prefix))
    each.append('{}tsne_2d_30p.npz'.format(prefix))
    each.append('{}tsne_2d_50p.npz'.format(prefix))
    each.append('{}tsne_2d_5p_poly.npz'.format(prefix))
    each.append('{}tsne_2d_10p_poly.npz'.format(prefix))
    each.append('{}tsne_2d_15p_poly.npz'.format(prefix))
    each.append('{}tsne_2d_30p_poly.npz'.format(prefix))
    each.append('{}tsne_2d_50p_poly.npz'.format(prefix))
    each.append('{}tsne_3d_30p.npz'.format(prefix))
    while True:
        for item in each:
            if not os.path.isfile(item):
                time.sleep(60)
                continue
        break
    selected = [np.load(each[i]) for i in selection]
    X_train = np.concatenate([item['train'] for item in selected], axis=1)
    X_valid = np.concatenate([item['valid'] for item in selected], axis=1)
    X_test = np.concatenate([item['test'] for item in selected], axis=1)
    np.savez('{}tsne.npz'.format(prefix), X_train=X_train, X_valid=X_valid, X_test=X_test)

def announce(text):
    sys.stdout.write('{}\n'.format('-' * 80))
    sys.stdout.write('{}\n'.format(text))
    sys.stdout.write('{}\n'.format('-' * 80))
    sys.stdout.flush()

def main():
    operation = os.getenv('OPERATION', 'All')
    if operation in ['LoadData', 'All']:
        os.system('curl -sL https://gitlab.com/altermarkive/Numerai-Tournament-Data-Sets/raw/master/load.sh | bash -s -- /workspace/input')
        try:
            os.mkdir('/workspace/output')
        except:
            pass
    if operation in ['PrepareData', 'All']:
        announce('Data Preparation')
        os.system('python3 /code/prep_data.py')
    if operation in ['LogisticRegression', 'All']:
        announce('Simple Logistic Regression')
        os.system('python3 /code/models/pipeline/simple.py')
    if operation in ['tSNE2D', 'All']:
        announce('t-SNE 2D')
        subprocess.Popen(['python3', '/code/fit_tsne.py'])
    if operation in ['tSNE3D', 'All']:
        announce('t-SNE 3D')
        subprocess.Popen(['python3', '/code/fit_tsne_3d.py'])
    if operation in ['tSNESummary', 'All']:
        announce('t-SNE Summary')
        merge_tsne([1])
    if operation in ['TFNN', 'All']:
        announce('TF NN')
        os.system('python3 /code/models/classifier/main.py')
    if operation in ['BasicVisualization', 'All']:
        announce('Basic data visualization notebook')
        os.system('python3 /code/notebooks/numerai.py')
    if operation in ['AdditionalVisualization', 'All']:
        announce('Additional data visualization notebook')
        os.system('python3 /code/notebooks/visualization.py')
    if operation in ['TFAutoencoder', 'All']:
        announce('TF Autoencoder')
        os.system('python3 /code/models/autoencoder/main.py')
    if operation in ['TFAdversarial', 'All']:
        announce('TF Adversarial')
        os.system('python3 /code/models/adversarial/main.py')
    if operation in ['TFPairwise', 'All']:
        announce('TF Pairwise')
        os.system('python3 /code/models/pairwise/main.py')
    if operation in ['Pairwise', 'All']:
        announce('Pairwise Interactions')
        os.system('python3 /code/models/pipeline/pairwise.py')
    if operation in ['ParameterSearch', 'All']:
        announce('Searching parameters')
        os.system('python3 /code/search_params.py')
    if operation in ['AdditionalLogisticRegression', 'All']:
        announce('Logistic Regression')
        os.system('python3 /code/models/pipeline/lr.py')
    if operation in ['FactorizationMachines', 'All']:
        announce('Factorization Machines')
        os.system('python3 /code/models/pipeline/fm.py')
    if operation in ['GradientBoostingTrees', 'All']:
        announce('GBT')
        os.system('python3 /code/models/pipeline/gbt.py')
    if operation in ['Ensemble', 'All']:
        announce('Ensemble')
        os.system('python3 /code/ensemble.py')
    if operation in ['TPOT', 'All']:
        announce('TPOT')
        os.system('python3 /code/tpot_test.py')

if __name__ == '__main__':
    main()
