# !/home/lxy/anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/7 上午10:46
# @Author  : lxy
# @FileName: utils.py
# @Software: PyCharm

import re
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import wordnet


# 探索类别的分布
def data_exploration(file):
    contents = pd.read_table(file)
    labels = contents.iloc[:,1].to_list()
    undep = set(labels)
    indexlist = range(0, len(undep))
    id2label = dict(zip(indexlist, undep))
    label2id = dict(zip(undep, indexlist))
    labels_index = [label2id[label] for label in labels]
    result = {}
    for i in set(labels_index):
        result[i] = labels_index.count(i)
    return result

# 使用Wordnet增强数据
def data_argument(items):
    pass


def pos_encoding(items, vocab_size,max_len=30):
    '''

    :param items: data_item list
    :param vocab_size
    :return:
    '''
    pos_embedding = np.ones(vocab_size, 4)
    for i, item in zip(range(0, len(items)), items):
        pos_embedding[i][0] = i
        pos_embedding[i][1] = i - item.sub[0]
        pos_embedding[i][2] = i - item.obj[0]
        pos_embedding[i][3] = item.sub[0] - item.obj[0]
    pos_embedding /= len(items)
    return pos_embedding




if __name__ == '__main__':
    '''
    {0: 1003, 1: 716, 2: 1410, 3: 634, 4: 940, 5: 540, 6: 690, 7: 845, 8: 504, 9: 717}
    没有关系的类别有1003个 其余的类别分布较为不均匀
    因此想到使用 EDA中的同义词替换 扩充数据
    '''
   # print(data_exploration('../datas/SemEval2010-Task8/train/train_result.txt'))
    print(wordnet.synsets('start'))  # 同义词替换的问题 词性


