# -*- coding: utf-8 -*-
# @Time    : 2018/4/20 10:29
# @Author  : Drxan
# @Email   : yuwei8905@126.com
# @File    : data.py
# @Software: PyCharm

import string
import sys
import warnings
from collections import OrderedDict, Counter
from hashlib import md5

import numpy as np
import pandas as pd
from six.moves import range
from six.moves import zip
from zhon.hanzi import punctuation as ch_punctuation
from string import punctuation as en_punctuation
import jieba
import re
import os


def load_stop_words(file_path, encoding='utf-8'):
    with open(file_path, encoding=encoding) as f:
        stop_words = [c.strip() for c in f.readlines()]
    return stop_words


def filter_text(text, filters=None, lower=True):
    """
    对文本进行过滤
    :param text: 需要过滤处理的文本
    :param filters: 需要过滤掉的字符集
    :param lower: 是否忽略英文字符的大小写
    :return: 过滤后的文本
    """
    if lower:
        text = text.lower()
    if filters is None:
        filters = ch_punctuation+en_punctuation+r'\t\r\n\f\v'
    sub_parttern = '['+filters+']'
    text = re.sub(sub_parttern, '', text)
    return text


def load_data(data_path, do_filter=True, filter=None, encoding='gb2312'):
    """
    加载原始数据
    :param data_path: 数据路径
    :param do_filter: 是否对原始数据进行过滤
    :param filter: 需要过滤的字符，只有do_filter为True时有效
    :param encoding: 原始文本数据的编码格式
    :return: x为原始文本列表，y为各文本对应的类别
    """
    x = []
    y = []
    categories = os.listdir(data_path)
    for cat_name in categories:
        cat_name = cat_name.strip()
        cat_path = os.path.join(data_path, cat_name)
        if os.path.isdir(cat_path):
            files = os.listdir(cat_path)
            for file in files:
                file_path = os.path.join(cat_path, file)
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    text = f.read()
                    if do_filter:
                        text = filter_text(text, filter)
                x.append(text)
                y.append(cat_name)
    return x, y


def text_to_token(texts, stop_words=None):
    """
    将原始文本进行分词，并统计每个次的频数
    :param texts: 原始文本列表
    :param stop_words: 停用词列表
    :return: 分词后的文本列表
    """
    word_counts = Counter()
    text_tokens = []
    for txt in texts:
        tokens = list(jieba.cut(txt, cut_all=False))
        word_counts.update(tokens)
        text_tokens.append(tokens)
    del word_counts['']
    del word_counts[' ']
    if stop_words is not None:
        for w in stop_words:
            del word_counts[w]
    return text_tokens, word_counts


def text_to_sequence(texts, stop_words=None, max_cnt=None, min_cnt=0):
    word_index = {}
    text_idx = []
    text_tokens, word_counts = text_to_token(texts, stop_words)
    if max_cnt is not None:
        for k in word_counts:
            if word_counts[k] > max_cnt:
                del word_counts[k]
    min_cnt = max(1, min_cnt)
    for text in text_tokens:
        idx = []
        for token in text:
            if word_counts[token] < min_cnt:
                continue
            if token not in word_index:
                word_index[token] = len(word_index)+1
            idx.append(word_index[token])
        text_idx.append(idx)
    return text_idx, word_counts, word_index


def convert_label_to_index(raw_labels):
    label_dict = {}
    labels = []
    for label_name in raw_labels:
        if label_name not in label_dict:
            label_dict[label_name] = len(label_dict)
            labels.append(label_dict[label_name])
    return labels, label_dict


