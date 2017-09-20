import glob
import os
import shutil
import sys
import numpy as np
import os

def merge_tsne(selection):
    prefix = os.getenv('PREFIX', '/workspace/output/')
    each = []
    each.append(np.load('{}tsne_2d_5p.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_10p.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_15p.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_30p.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_50p.npz'.format(prefix)))
    each.append(np.load('{}tsne_3d_30p.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_5p_poly.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_10p_poly.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_15p_poly.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_30p_poly.npz'.format(prefix)))
    each.append(np.load('{}tsne_2d_50p_poly.npz'.format(prefix)))
    each.append(np.load('{}tsne_3d_30p.npz'.format(prefix)))
    selected = [each[i] for i in selection]
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
    if operation in ['t-SNE', 'All']:
        prefix = os.getenv('PREFIX', '/workspace/output/')
        announce('t-SNE Python')
        os.system('python3 /code/fit_tsne.py')
        os.rename('{}tsne_3d_30p.npz'.format(prefix), '{}tsne_3d_30p_tsne.npz'.format(prefix))
        announce('t-SNE C')
        os.system('python3 /code/bh_tsne/prep_data.py')
        os.system('/code/bh_tsne/bh_tsne')
        os.system('python3 /code/bh_tsne/prep_result.py')
        os.rename('{}tsne_3d_30p.npz'.format(prefix), '{}tsne_3d_30p_bhtsne.npz'.format(prefix))
        announce('tSNE selection')
        try:
            os.remove('{}tsne_3d_30p.npz'.format(prefix))
        except:
            pass
        shutil.copyfile('{}tsne_3d_30p_bhtsne.npz'.format(prefix), '{}tsne_3d_30p.npz'.format(prefix))
        announce('t-SNE Merge')
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
