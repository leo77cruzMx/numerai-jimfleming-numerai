import os

if __name__ == '__main__':
    perplexity = 30
    os.system('python3 /code/bh_tsne/prep_data.py {}'.format(perplexity))
    os.system('/code/bh_tsne/bh_tsne')
    os.system('python3 /code/bh_tsne/prep_result.py {}'.format(perplexity))
