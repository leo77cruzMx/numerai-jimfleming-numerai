import glob
import os
import shutil

def main():
    if not os.path.isdir('/data'):
        os.mkdir('/data')
    if os.path.isdir('/data/.git'):
        os.chdir('/data')
        os.system('git pull')
        os.chdir('/')
    else:
        os.system('git clone https://github.com/altermarkive/Numerai-Tournament-Data-Sets.git /data')
    last = sorted([item for item in os.listdir('/data') if item.isdigit()])[-1]
    try:
        os.mkdir('/data/round')
    except:
        pass
    for item in glob.glob('/data/%s/*' % last):
        shutil.copy(item, '/data/round/')
    try:
        os.mkdir('/predictions')
    except:
        pass
    print('Data Preparation')
    os.system('python3 /code/prep_data.py')
    print('Feature Engineering')
    os.system('python3 /code/fit_tsne.py')
    print('Simple Logistic Regression')
    os.system('python3 /code/models/pipeline/simple.py')
    print('Pairwise Interactions')
    os.system('python3 /code/models/pipeline/pairwise.py')
    print('Logistic Regression')
    os.system('python3 /code/models/pipeline/lr.py')
    print('Factorization Machines')
    os.system('python3 /code/models/pipeline/fm.py')
    print('Ensemble')
    os.system('python3 /code/ensemble.py')

if __name__ == '__main__':
    main()
