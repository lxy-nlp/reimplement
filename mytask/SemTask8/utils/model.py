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
import numpy as np

from mytask.SemTask8.utils import constant


class SimpleModel(nn.Module):
    '''
    准备使用args一个obj替换 下面的参数
    '''
    def __init__(self, vocab_size, embedding_size, pre_embeddings, pos_dim, hidden_dim, head, num_class, pos_need, drop_out, bidirection=False):
        ''''''
        super(SimpleModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings.weight.data.copy_(torch.from_numpy(pre_embeddings))
        self.direction = 2 if bidirection is True else 1
        self.pos_need = pos_need
        self.pos_conv = nn.Linear(pos_dim, pos_dim * 5)
        self.hidden_dim = self.direction * hidden_dim
        self.head = head
        self.drop_out = nn.Dropout(p=0.3)
        self.pos_dim = 0 if pos_need == False else pos_dim * 5
        self.lstm = nn.LSTM(embedding_size + self.pos_dim, hidden_dim, bidirectional=bidirection)
        self.attn = MultiHeadAttention(head, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(self.hidden_dim, num_class)

    def forward(self, sentence_ids, pos_emb, sub_pos, obj_pos, mask_matrix):
        '''
        :param sentence_ids:
        :param pos_emb:
        :param mask_matrix:
        :return:
        output(seq_len, batch, hidden_size * num_directions)
        hn(num_layers * num_directions, batch, hidden_size)
        cn(num_layers * num_directions, batch, hidden_size)
        在这里应该使用的是output
        '''
        x_emb = self.embeddings(torch.tensor(sentence_ids))
        batch,sen_len, hidden_dim = x_emb.shape
        pos_emb = torch.tensor(pos_emb, dtype=torch.float32)
        pos_emb = self.pos_conv(pos_emb)
        x_emb = torch.cat((x_emb, pos_emb), -1) if self.pos_need else x_emb
        out, (h_n, c_n) = self.lstm(x_emb)
        attn_tensor = self.attn(out, out)
        sentence_list = torch.matmul(attn_tensor, out.reshape(batch, self.head, sen_len, self.hidden_dim // self.head))  # 得出加入权重的后的隐状态
        sentence_list = sentence_list.reshape(batch, sen_len, -1)
        sentence_list = self.drop_out(sentence_list)
        sub_pos = torch.tensor(sub_pos, dtype=torch.float32)
        obj_pos = torch.tensor(obj_pos, dtype=torch.float32)
        pool_mask = torch.tensor(mask_matrix, dtype=torch.float32)
        sub_mask, obj_mask, pool_mask = sub_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2), pool_mask.eq(0).eq(0).unsqueeze(2)
        h_out = pool(sentence_list, pool_mask, type="avg")
        subj_out = pool(sentence_list, sub_mask, type="max")
        obj_out = pool(sentence_list, obj_mask, type="max")
        relation = torch.cat((subj_out, h_out, obj_out), dim=-1)
        res = self.classifier(self.activation(self.linear(relation)))
        return res
        '''
         # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type="max")
        obj_out = pool(h, obj_mask, type="max")
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        '''
        '''
        tensor([[[-0.1538,  0.5741, -1.8462, -0.9306],
         [ 1.7812,  1.4193, -0.5302, -1.0420],
         [ 0.6882, -0.7270, -1.8343,  0.9276]],

        [[ 0.9223,  0.7743,  1.1231,  0.1709],
         [-1.1375,  0.7753, -0.4947, -1.0745],
         [-1.3858,  0.0279,  0.4381, -0.4061]]])
         
        
        tensor([[[-0.1538,  0.5741, -1.8462, -0.9306],
         [ 1.7812,  1.4193, -0.5302, -1.0420]],

        [[ 0.6882, -0.7270, -1.8343,  0.9276],
         [ 0.9223,  0.7743,  1.1231,  0.1709]],

        [[-1.1375,  0.7753, -0.4947, -1.0745],
         [-1.3858,  0.0279,  0.4381, -0.4061]]])

        '''

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

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

    def __init__(self, h, d_model, dropout=0.2):
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









