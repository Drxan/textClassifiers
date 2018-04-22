# -*- coding: utf-8 -*-
# @Time    : 2018/4/20 10:56
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : train.py
# @Software: PyCharm

from drxan import preprocessing as pp
import os
import time
import pandas as pd
import numpy as np
import json

current_path = os.getcwd()
data_dirs = dict()
data_dirs['token_count'] = os.path.join(current_path, 'data/token_count.csv')
data_dirs['token_dict'] = os.path.join(current_path, 'data/token_dict.txt')
data_dirs['text_sequence'] = os.path.join(current_path, 'data/text_sequence.txt')
data_dirs['label_dict'] = os.path.join(current_path, 'data/label_dict.txt')
data_dirs['labels'] = os.path.join(current_path, 'data/labels.txt')
data_dirs['train_data'] = os.path.join(current_path, 'data/train')
data_dirs['test_data'] = os.path.join(current_path, 'data/test')
data_dirs['stop_words'] = os.path.join(current_path, 'data/stop_words.txt')


def prepare_data():
    print('[1] Loading raw datas...')
    texts, labels = pp.load_data(data_dirs['train_data'], do_filter=True, filter=None, encoding='gb2312')
    labels, label_dict = pp.convert_label_to_index(labels)
    stop_words = pp.load_stop_words(data_dirs['stop_words'])

    print('[2] Converting raw texts to tokens denoted by integers...')
    start = time.time()
    texts_idx, token_counts, token_dict = pp.text_to_sequence(texts, stop_words=stop_words)
    token_dict['UNK'] = len(token_dict)+1
    print('Time:', time.time()-start)

    print('[3] Saving token counts to file...')
    token_counts = pd.Series(token_counts).sort_values(ascending=False)
    if os.path.exists(data_dirs['token_count']):
        os.remove(data_dirs['token_count'])
    token_counts.to_csv(data_dirs['token_count'])

    print('[4] Saving token dict to file...')
    with open(data_dirs['token_dict'], 'w') as f:
        f.write(json.dumps(token_dict))

    print('[5] Saving texts denoted by token indexes...')
    with open(data_dirs['text_sequence'], 'w') as f:
        f.write('\n'.join([','.join([str(token) for token in text]) for text in texts_idx]))

    print('[6] Saving text label to file...')
    with open(data_dirs['labels'], 'w') as f:
        f.write('\n'.join([str(label) for label in labels]))

    print('[7] Saving label dict to file...')
    with open(data_dirs['label_dict'], 'w') as f:
        f.write(json.dumps(label_dict))

    return texts_idx, labels


if __name__ == '__main__':
    x, y = prepare_data()
    lens = [len(t) for t in x]
    print('Mean leangth:', np.mean(lens))
    print('Max leangth:', np.max(lens))
    print('Min leangth:', np.min(lens))
    print('Median leangth:', np.median(lens))