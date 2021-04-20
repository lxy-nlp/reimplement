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
import torch.utils.data as Data
import torch
import random

from mytask.SemTask8.utils import constant

random.seed(1234)
np.random.seed(1234)

class Vocab:
    def __init__(self, filename, wv_dim, load=False, word_counter=None, threshold=0):
        if load:
            assert os.path.exists(filename),"Vocab file does not exist at " + filename

            self.id2word, self.word2id, self.emdeddings = self.load_glove_vocab(filename, wv_dim)
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
        emb[constant.PAD_ID] = [0]*wv_dim  # <pad> should be all 0
        emb[constant.UNK_ID] = [1]*wv_dim
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
    def __init__(self, sub, obj, sentence, label, label_id):
        self.sub = sub
        self.obj = obj
        self.sentence = sentence
        self.sent_ids = []
        self.label = label
        self.label_id = label_id
        self.pos_encoding = None
        self.mask = None

    def map(self, token_list, word2id):
        """
        Map a list of tokens to their ids.
        """
        self.sent_ids = [word2id[w] if w in word2id else constant.UNK_ID for w in token_list]
        return self.sent_ids

    # 位置编码
    def pos_encode(self):
        if len(self.sentence) < 1:
            raise NotImplementedError
        pos_embedding = np.ones((len(self.sentence), 4))
        for i, item in zip(range(0, len(self.sentence)), self.sentence):
            pos_embedding[i][0] = i
            pos_embedding[i][1] = i - self.sub[0]
            pos_embedding[i][2] = i - self.obj[0]
            pos_embedding[i][3] = self.sub[0] - self.obj[0]
        pos_embedding /= len(self.sentence)
        self.pos_encoding = pos_embedding
        return pos_embedding

    # 从sequence_list中挖去  主体 和 客体
    def mask_entity(self):
        mask_matrix = [1] * len(self.sentence)
        sub_mask = [0] * len(self.sentence)
        obj_mask = [0] * len(self.sentence)
        mask_matrix[self.sub[0]:self.sub[1]+1] = [0] * (self.sub[1] - self.sub[0] + 1)
        mask_matrix[self.obj[0]:self.obj[1]+1] = [0] * (self.obj[1] - self.obj[0] + 1)
        sub_mask[self.sub[0]:self.sub[1]+1] = [1] * (self.sub[1] - self.sub[0] + 1)
        obj_mask[self.obj[0]:self.obj[1]+1] = [1] * (self.obj[1] - self.obj[0] + 1)
        return mask_matrix, sub_mask, obj_mask

def read_labels(path):
    '''
    获取全部的标签
    :param path:
    :return: 原始标签和标签id
    '''
    contents = pd.read_table(path,header=None)
    labels = contents.iloc[:, 1].to_list()
    undep = set(labels)
    indexlist = range(0, len(undep))
    id2label = dict(zip(indexlist, undep))
    label2id = dict(zip(undep, indexlist))
    labels_index = [label2id[label] for label in labels]
    return labels, labels_index

def read_datas(path, word2id, labels, labels_index, max_len=70):
    '''
    用于处理数据
    :param path: 数据集路径
    :return: 处理后的数据
    处理后的数据格式 [obj,sub.sentence]
    '''
    with open(path, 'r') as f:
        lines = f.readlines()

    all_items = []
    all_pos = []
    items = []
    for i, line in enumerate(lines):
        single_item = []
        line_split = line.split()
        sub = []  # 主体的 首尾位置
        obj = []  # 客体的 首尾位置
        for i, item in zip(range(0, len(line_split)), line_split):
            if '<e1>' in item:
                sub.append(i)
                # all_sub_start.append(i)
            if '</e1>' in item:
                sub.append(i)
                # all_sub_end.append(i)
            if '<e2>' in item:
                obj.append(i)
                # all_obj_start.append(i)
            if '</e2>' in item:
                obj.append(i)
                # all_obj_end.append(i)
        line = re.sub(r'<e.>', '', line)
        line = re.sub(r'<.e.>', '', line)
        sequence = line.split()
        sequence = sequence[:max_len] if len(sequence) >= max_len else sequence + ['PAD'] * (max_len - len(sequence))
        # sequence = sequence[:max_len] if len(sequence) > max_len else sequence
        one_item = data_item(sub, obj, sequence, labels[i], labels_index[i])
        sentence_ids = one_item.map(sequence, word2id)
        mask_matrix, sub_mask, obj_mask = one_item.mask_entity()
        pos_embedding = one_item.pos_encode()
        single_item.append(sentence_ids)
        single_item.append(sub_mask)
        single_item.append(obj_mask)
        single_item.append(mask_matrix)
        all_pos.append(pos_embedding)
        all_items.append(single_item)
        items.append(one_item)
    return all_items, labels_index, all_pos,items  # 返回值是 训练数据 和 标签

def load_data(batch_size,items,labels,all_pos):
    # shuffle  打乱数据为了 缓解过拟合
    indexes = [i for i in range(len(labels))]
    random.shuffle(indexes)
    items = np.array(items)[indexes].tolist()
    labels = np.array(labels)[indexes].tolist()
    all_pos = np.array(all_pos)[indexes].tolist()
    item_batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    label_batches = [labels[i:i+batch_size] for i in range(0, len(labels), batch_size)]
    pos_bathes = [all_pos[i:i+batch_size] for i in range(0, len(all_pos), batch_size)]

    return item_batches, label_batches, pos_bathes

# 位置编码
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

if __name__ == '__main__':
      vocab =  Vocab('../../datas/pre_embeddings/glove/glove.6B.200d.txt', 200, True)
      labels, labels_index = read_labels('../../datas/SemEval2010-Task8/train/train_result.txt')
      all_items, labels = read_datas('../../datas/SemEval2010-Task8/train/train.txt',vocab.word2id,labels,labels_index)
      item_batches, label_batches = load_data(50, all_items, labels)
