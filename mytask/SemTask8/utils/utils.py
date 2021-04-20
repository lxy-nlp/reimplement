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
from mytask.SemTask8.utils import constant
from nltk.corpus import wordnet

from mytask.SemTask8.utils.vocab import data_item


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

def sequence_length(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    all_length = []
    for i, line in enumerate(lines):
        line_split = line.split()
        all_length.append(len(line_split))
    return all_length

# 使用Wordnet增强数据
def data_argument(items,counts,types,word2id):
    '''

    :param items: 全部数据
    :param counts: 扩充后的数量
    :param type: 需要扩充的类型 list [1,2,3,6,7]
    :return: 增加后的数据
    '''
    added_item = []
    added_label = []
    added_pos = []
    for item in items:
        if (item.label in types) and (item.sub[0] == item.sub[1]) and (item.obj[0] == item.obj[1]):
            sub_new = wordnet.synsets(item.sub[0])[0].lemma_names[0]
            obj_new = wordnet.synsets(item.obj[0])[0].lemma_names[0]
            new_item = data_item(item.sub, item.obj, item.sentence, item.label, item.label_id)
            new_item.sentence[item.sub] = sub_new
            new_item.sentence[item.obj] = obj_new
            new_item.map(new_item.sentence, word2id)
            added_item.append(new_item)
            added_pos.append(new_item.pos_encode())
            added_label.append(item.label)
    return added_item, added_label, added_pos



def pos_encoding(items, vocab_size):
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

def evaluate_batch_based(predicted_batch, gold_batch, threshold = 1.0, idx2label=None, empty_label = None):
    if len(predicted_batch) != len(gold_batch):
        raise TypeError("predicted_idx and gold_idx should be of the same length.")

    correct = 0
    for i in range(len(gold_batch)):
        rec_batch = micro_avg_precision(predicted_batch[i], gold_batch[i], empty_label)
        if rec_batch >= threshold:
            correct += 1

    acc_batch = correct / float(len(gold_batch))

    return acc_batch


def evaluate_instance_based(predicted_idx, gold_idx, idx2label=None, empty_label=None):
    if len(predicted_idx) != len(gold_idx):
        raise TypeError("predicted_idx and gold_idx should be of the same length.")
    if idx2label:
        label_y = [idx2label[element] for element in gold_idx]
        pred_labels = [idx2label[element] for element in predicted_idx]
    else:
        label_y = gold_idx
        pred_labels = predicted_idx

    prec = micro_avg_precision(pred_labels, label_y, empty_label)
    rec = micro_avg_precision(label_y, pred_labels, empty_label)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1


def micro_avg_precision(guessed, correct, empty=None):
    """
    Tests:
    >>> micro_avg_precision(['A', 'A', 'B', 'C'],['A', 'C', 'C', 'C'])
    0.5
    >>> round(micro_avg_precision([0,0,0,1,1,1],[1,0,1,0,1,0]), 6)
    0.333333
    """
    correctCount = 0
    count = 0

    idx = 0
    while idx < len(guessed):
        if guessed[idx] != empty:
            count += 1
            if guessed[idx] == correct[idx]:
                correctCount += 1
        idx += 1
    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision

if __name__ == '__main__':
    '''
    {0: 1003, 1: 716, 2: 1410, 3: 634, 4: 940, 5: 540, 6: 690, 7: 845, 8: 504, 9: 717}
    没有关系的类别有1003个 其余的类别分布较为不均匀
    因此想到使用 EDA中的同义词替换 扩充数据
    '''
    all_length = sequence_length('../datas/SemEval2010-Task8/train/train.txt')
    print(max(all_length), min(all_length))
    print(data_exploration('../datas/SemEval2010-Task8/train/train_result.txt'))
   # print(wordnet.synsets('start'))  # 同义词替换的问题 词性

