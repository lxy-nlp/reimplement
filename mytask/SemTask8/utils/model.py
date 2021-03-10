# !/home/lxy/anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/7 下午3:46
# @Author  : lxy
# @FileName: model.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class  SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, max_len,num_class,bidirection=False):
        ''''''
        super(SimpleModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_dim,bidirectional=bidirection)
        self.direction = 2 if bidirection==True else 1
        self.linear1 = nn.Linear(hidden_dim * self.direction, hidden_dim//2)
        self.activate = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim // 2, num_class)
        self.attn = MultiHeadAttention()

    def forward(self, x):
        '''

        :param x:
        :return:
        output(seq_len, batch, hidden_size * num_directions)
        hn(num_layers * num_directions, batch, hidden_size)
        cn(num_layers * num_directions, batch, hidden_size)

        '''
        x_emb = self.embbeddings(x)
        out,(h_n,c_n) = self.lstm(x_emb)







# 多头注意力机制

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn









