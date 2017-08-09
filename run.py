import glob
import os
import shutil
import sys
import numpy as np

def merge_tsne(selection):
    each = []
    each.append(np.load('/output/tsne_2d_5p.npz'))
    each.append(np.load('/output/tsne_2d_10p.npz'))
    each.append(np.load('/output/tsne_2d_15p.npz'))
    each.append(np.load('/output/tsne_2d_30p.npz'))
    each.append(np.load('/output/tsne_2d_50p.npz'))
    each.append(np.load('/output/tsne_3d_30p.npz'))
    each.append(np.load('/output/tsne_2d_5p_poly.npz'))
    each.append(np.load('/output/tsne_2d_10p_poly.npz'))
    each.append(np.load('/output/tsne_2d_15p_poly.npz'))
    each.append(np.load('/output/tsne_2d_30p_poly.npz'))
    each.append(np.load('/output/tsne_2d_50p_poly.npz'))
    each.append(np.load('/output/tsne_3d_30p.npz'))
    selected = [each[i] for i in selection]
    X_train = np.concatenate([item['train'] for item in selected], axis=1)
    X_valid = np.concatenate([item['valid'] for item in selected], axis=1)
    X_test = np.concatenate([item['test'] for item in selected], axis=1)
    np.savez('/output/tsne.npz', X_train=X_train, X_valid=X_valid, X_test=X_test)

def announce(text):
    sys.stdout.write('{}\n'.format('-' * 80))
    sys.stdout.write('{}\n'.format(text))
    sys.stdout.write('{}\n'.format('-' * 80))
    sys.stdout.flush()

def main():
    os.system('curl -sL https://raw.githubusercontent.com/altermarkive/Numerai-Tournament-Data-Sets/master/load.sh | bash -s -- /input')
    try:
        os.mkdir('/output')
    except:
        pass
    announce('Data Preparation')
    os.system('python3 /code/prep_data.py')
    announce('Simple Logistic Regression')
    os.system('python3 /code/models/pipeline/simple.py')
    announce('t-SNE')
    os.system('python3 /code/fit_tsne.py')
    os.rename('/output/tsne_3d_30p.npz', '/output/tsne_3d_30p_tsne.npz')
    announce('t-SNE C')
    os.system('python3 /code/bh_tsne/prep_data.py')
    os.system('/code/bh_tsne/bh_tsne')
    os.system('python3 /code/bh_tsne/prep_result.py')
    os.rename('/output/tsne_3d_30p.npz', '/output/tsne_3d_30p_bhtsne.npz')
    announce('t-SNE selection')
    try:
        os.remove('/output/tsne_3d_30p.npz')
    except:
        pass
    shutil.copyfile('/output/tsne_3d_30p_bhtsne.npz', '/output/tsne_3d_30p.npz')
    announce('TF NN')
    os.system('python3 /code/models/classifier/main.py')
    announce('Basic data visualization notebook')
    os.system('runipy /code/notebooks/Numerai.ipynb /code/notebooks/NumeraiOutput.ipynb')
    os.system('jupyter nbconvert --to html /code/notebooks/NumeraiOutput.ipynb')
    shutil.move('/code/notebooks/NumeraiOutput.ipynb', '/output/Numerai.ipynb')
    shutil.move('/code/notebooks/NumeraiOutput.html', '/output/Numerai.html')
    announce('Additional data visualization notebook')
    os.system('runipy /code/notebooks/Visualization.ipynb /code/notebooks/VisualizationOutput.ipynb')
    os.system('jupyter nbconvert --to html /code/notebooks/VisualizationOutput.ipynb')
    shutil.move('/code/notebooks/VisualizationOutput.ipynb', '/output/Visualization.ipynb')
    shutil.move('/code/notebooks/VisualizationOutput.html', '/output/Visualization.html')
    shutil.move('/code/notebooks/tsne-3d-scatter.html', '/output/tsne-3d-scatter.html')
    announce('TF Autoencoder')
    os.system('python3 /code/models/autoencoder/main.py')
    announce('TF Adversarial')
    os.system('python3 /code/models/adversarial/main.py')
    announce('TF Pairwise')
    merge_tsne([1])
    os.system('python3 /code/models/pairwise/main.py')
    announce('Pairwise Interactions')
    os.system('python3 /code/models/pipeline/pairwise.py')
    announce('Searching parameters')
    os.system('python3 /code/search_params.py')
    announce('Logistic Regression')
    os.system('python3 /code/models/pipeline/lr.py')
    announce('Factorization Machines')
    os.system('python3 /code/models/pipeline/fm.py')
    announce('GBT')
    os.system('python3 /code/models/pipeline/gbt.py')
    announce('Ensemble')
    os.system('python3 /code/ensemble.py')
    announce('TPOT')
    os.system('python3 /code/tpot_test.py')

if __name__ == '__main__':
    main()
