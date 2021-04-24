# !/home/lxy/anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/21 下午7:23
# @Author  : lxy
# @FileName: test.py
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
from mytask.SemTask8.utils.model import SimpleModel
from mytask.SemTask8.utils.utils import evaluate_instance_based
from mytask.SemTask8.utils.vocab import Vocab, read_labels, read_datas, load_data


def test(model, item_batches, label_batches, pos_batches, device):
    model.eval()
    truth_labels = []
    pre_labels = []
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    count = 0
    for (batchx, batchy), pos in zip(zip(item_batches, label_batches), pos_batches):
        sentence_ids = torch.tensor(list(np.array(batchx)[:, 0])).to(device)
        sub_pos = torch.tensor(list(np.array(batchx)[:, 1]), dtype=torch.float32).to(device)
        obj_pos = torch.tensor(list(np.array(batchx)[:, 2]), dtype=torch.float32).to(device)
        mask_matrix = torch.tensor(list(np.array(batchx)[:, 3]), dtype=torch.float32).to(device)
        pos_emb = torch.tensor(list(np.array(pos)), dtype=torch.float32).to(device)
        result_class, attn, pos_weight = model(sentence_ids, pos_emb, sub_pos, obj_pos, mask_matrix)
        batchy = torch.tensor(batchy, dtype=torch.int64)
        _, predict = torch.max(result_class.cpu(), dim=1)
        labels = batchy.reshape(-1).tolist()  # label是输入标签的id值
        predict = predict.data.tolist()
        predict = np.array(predict).tolist()
        labels = np.array(labels).tolist()
        truth_labels += labels
        pre_labels += predict
        val_loss += criterion(result_class.cpu(), batchy) + 0.2 * torch.norm(attn.cpu(), p='fro') # + 0.1 * torch.norm(pos_weight, p='fro')  # label_y
        count += 1
    if(len(pre_labels) != len(truth_labels)):
        raise Exception
    return pre_labels, truth_labels, val_loss/count

'''
def test(model,items, labels, poses):
    model.eval()
    truth_labels = []
    pre_labels = []

    sentence_ids = list(np.array(items)[:, 0])
    sub_pos = list(np.array(items)[:, 1])
    obj_pos = list(np.array(items)[:, 2])
    mask_matrix = list(np.array(items)[:, 3])
    pos_emb = list(np.array(poses))
    result_class = model(sentence_ids, pos_emb, sub_pos, obj_pos, mask_matrix)
    batchy = torch.tensor(labels, dtype=torch.int64)
    _, predict = torch.max(result_class, dim=1)
    labels = batchy.reshape(-1).tolist()  # label是输入标签的id值
    predict = predict.data.tolist()
    predict = np.array(predict).tolist()
    labels = np.array(labels).tolist()
    truth_labels += labels
    pre_labels += predict

    if(len(pre_labels) != len(truth_labels)):
        raise Exception
    p,r,f1 = evaluate_instance_based(pre_labels,truth_labels,empty_label=0)
    return p,r,f1
'''
