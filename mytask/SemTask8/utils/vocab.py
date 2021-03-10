# !/home/lxy/anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/7 下午3:05
# @Author  : lxy
# @FileName: vocab.py
# @Software: PyCharm

import os
import random
import numpy as np
import pandas as pd
import pickle
import re

from mytask.SemTask8.utils import constant

random.seed(1234)
np.random.seed(1234)

class Vocab:
    def __init__(self, filename, wv_dim, load=False, word_counter=None, threshold=0):
        if load:
            assert os.path.exists(filename),"Vocab file does not exist at " + filename

            self.id2word,self.word2id,self.emdeddings = self.load_glove_vocab(filename, wv_dim)
            self.size = len(self.id2word)

        else:
            print("Creating vocab from scratch...")
            assert word_counter is not None, "word_counter is not provided for vocab creation."
            self.word_counter = word_counter
            if threshold > 1:
                # remove words that occur less than thres
                self.word_counter = dict([(k,v) for k,v in self.word_counter.items() if v >= threshold])
            self.id2word = sorted(self.word_counter, key=lambda k:self.word_counter[k], reverse=True)
            # add special tokens to the beginning
            self.id2word = [constant.PAD_TOKEN, constant.UNK_TOKEN] + self.id2word
            self.word2id = dict([(self.id2word[idx],idx) for idx in range(len(self.id2word))])
            self.size = len(self.id2word)
            self.save(filename)

    def load(self, filename):
            with open(filename, 'rb') as infile:
                id2word = pickle.load(infile)
                word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])
            return id2word, word2id

    def load_glove_vocab(self, file, wv_dim):
        vocab = set()
        with open(file, encoding='utf8') as f:
            for line in f:
                elems = line.split()
                token = ''.join(elems[0:-wv_dim])
                vocab.add(token)
        id2word = {i+2: w for i, w in enumerate(vocab)}
        id2word[0] = '<PAD>'
        id2word[1] = '<UNK>'
        word2id = {w : i for i, w in zip(id2word.keys(),id2word.values())}
        vocab_size = len(vocab)
        emb = np.random.uniform(-1, 1, (vocab_size + 2, wv_dim))
        emb[constant.PAD_ID] = 0  # <pad> should be all 0
        emb[constant.UNK_ID] = 1
        with open(file, encoding="utf8") as f:
            for line in f:
                elems = line.split()
                token = ''.join(elems[0:-wv_dim])
                if token in word2id:
                    emb[word2id[token]] = [float(v) for v in elems[-wv_dim:]]
        return id2word, word2id, emb


    def save(self, filename):
            if os.path.exists(filename):
                print("Overwriting old vocab file at " + filename)
                os.remove(filename)
            with open(filename, 'wb') as outfile:
                pickle.dump(self.id2word, outfile)
            return

    def map(self, token_list):
        """
        Map a list of tokens to their ids.
        """
        return [self.word2id[w] if w in self.word2id else constant.UNK_ID for w in token_list]

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2word[idx] for idx in idx_list]


class data_item:
    def __init__(self, sub, obj, sentence,label,label_id):
        self.sub = sub
        self.obj = obj
        self.sentence = sentence
        self.sent_ids = []
        self.label = label
        self.label_id = label_id

    def map(self, token_list, word2id):
        """
        Map a list of tokens to their ids.
        """
        self.sent_ids = [word2id[w] if w in word2id else constant.UNK_ID for w in token_list]
        return self.sent_ids


def read_labels(path):
    '''
    获取全部的标签
    :param path:
    :return: 原始标签和标签id
    '''
    contents = pd.read_table(path)
    labels = contents.iloc[:, 1].to_list()
    undep = set(labels)
    indexlist = range(0, len(undep))
    id2label = dict(zip(indexlist, undep))
    label2id = dict(zip(undep, indexlist))
    labels_index = [label2id[label] for label in labels]
    return labels, labels_index

def read_datas(path, word2id,labels, labels_index):
    '''
    用于处理数据
    :param path: 数据集路径
    :return: 处理后的数据
    处理后的数据格式 [obj,sub.sentence]
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
    all_datas = []
    for i, line in enumerate(lines):
        line_split = line.split()
        sub = []  # 主体的 首尾位置
        obj = []  # 客体的 首尾位置
        for i, item in zip(range(0, len(line_split)), line_split):
            if '<e1>' in item:
                sub.append(i)
            if '</e1>' in item:
                sub.append(i)
            if '<e2>' in item:
                obj.append(i)
            if '</e2>' in item:
                obj.append(i)
        line = re.sub(r'<e.>', '', line)
        line = re.sub(r'<.e.>', '', line)
        sequence = line.split()
        one_item = data_item(sub, obj, sequence, labels[i],labels_index[i])
        one_item.map(sequence, word2id)
        all_datas.append(one_item)
    return all_datas




if __name__ == '__main__':
      vocab =  Vocab('../../datas/pre_embeddings/glove/glove.6B.200d.txt', 200, True)
      labels, labels_index = read_labels('../../datas/SemEval2010-Task8/train/train_result.txt')
      all_datas = read_datas('../../datas/SemEval2010-Task8/train/train.txt',vocab.word2id,labels,labels_index)
      print(all_datas)