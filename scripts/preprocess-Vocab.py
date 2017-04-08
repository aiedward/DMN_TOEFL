import os
import glob
import sys
import shutil
from utils import build_vocab
if __name__ == '__main__':
    print('=' * 80)
    print('Caculate the Vocab based on *.toks')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    sent_paths = list()
    query_paths = list()
    choice_paths = list()
    data_types = ['manual_trans']
    # data_types = ['manual_trans','ASR_trans']

    for data_type in data_types:
        toefltask_dir = os.path.join(data_dir, 'toefl',data_type)

        sent_paths.extend(glob.glob(os.path.join(toefltask_dir, '*/sents.toks')))
        query_paths.extend(glob.glob(os.path.join(toefltask_dir ,'*/queries_sep.toks')))
        choice_paths.extend(glob.glob(os.path.join(toefltask_dir,'*/choices.toks')))

    for data_type in data_types:
        toefltask_dir = os.path.join(data_dir, 'toefl',data_type)
        build_vocab(sent_paths+query_paths+choice_paths, os.path.join(toefltask_dir, 'vocab.txt'))
        build_vocab(sent_paths+query_paths+choice_paths, os.path.join(toefltask_dir, 'vocab-cased.txt'), lowercase=False)
