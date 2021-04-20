# !/home/lxy/anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/14 下午1:31
# @Author  : lxy
# @FileName: train.py
# @Software: PyCharm

import torch
import torch.nn as nn
import datetime
import numpy as np

from mytask.SemTask8.test import test
from mytask.SemTask8.utils.model import SimpleModel
from mytask.SemTask8.utils.utils import evaluate_instance_based, data_argument
from mytask.SemTask8.utils.vocab import Vocab, read_labels, read_datas,load_data


def train(model, epoches, item_batches, label_batches, pos_batches, test_item_batches, test_label_batches, test_pos_batches,learning_rate):
    optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, 50) # 余弦退火法优化策略
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoches):
        epoch_label = []
        epoch_pre = []
        for (batchx, batchy),pos in zip(zip(item_batches, label_batches), pos_batches):
            sentence_ids = list(np.array(batchx)[:, 0])
            sub_pos = list(np.array(batchx)[:, 1])
            # print([len(pos) for pos in sub_pos])
            obj_pos = list(np.array(batchx)[:, 2])
            # print([len(pos) for pos in obj_pos])
            mask_matrix = list(np.array(batchx)[:, 3])
            pos_emb = list(np.array(pos))
            result_class = model(sentence_ids, pos_emb, sub_pos, obj_pos, mask_matrix)
            optimizer_1.zero_grad()  # 清空梯度
            batchy = torch.tensor(batchy, dtype=torch.int64)
            train_loss = criterion(result_class, batchy)  # label_y
            train_loss.backward()  # 反向传播
            nn.utils.clip_grad_norm(model.parameters(), 0.15) # 梯度削减策略
            optimizer_1.step() #
            _,predict = torch.max(result_class, dim=1)
            labels = batchy.reshape(-1).tolist()    # label是输入标签的id值
            predict = predict.data.tolist()
            # p_indices = np.array(labels) != 0
            predict = np.array(predict).tolist()
            labels = np.array(labels).tolist()
            epoch_label += labels
            epoch_pre += predict
        # prcision, recall, add_f1 = evaluate_instance_based(epoch_pre, epoch_label)
        p, r, f1 = evaluate_instance_based(epoch_pre, epoch_label, empty_label=0)
        print('迭代次数为{},训练集预测精准率是{},召回率是{},f1是{}'.format(epoch, p, r, f1))
        test(model, test_item_batches, test_label_batches, test_pos_batches)
            # f1 += add_f1
        # print(prcision, recall, add_f1)



vocab = Vocab('../datas/pre_embeddings/glove/glove.6B.200d.txt', 200, True)
labels, labels_index = read_labels('../datas/SemEval2010-Task8/train/train_result.txt')

all_items, labels, all_pos, items = read_datas('../datas/SemEval2010-Task8/train/train.txt', vocab.word2id, labels, labels_index)
types = [1,2,3,6,7]
added_item, added_label,added_pos = data_argument(items, 0, types, vocab.word2id)
all_items += added_item
labels += added_label
all_pos += added_pos


item_batches, label_batches,pos_batches = load_data(50, all_items, labels,all_pos)
model = SimpleModel(400002, 200, vocab.emdeddings, 4, 200, 4, 10, False, True)

test_labels, test_labels_index = read_labels('../datas/SemEval2010-Task8/test/test_result.txt')
all_test_items, all_test_labels, all_test_pos,_ = read_datas('../datas/SemEval2010-Task8/test/test.txt', vocab.word2id, test_labels, test_labels_index)
test_item_batches, test_label_batches, test_pos_batches = load_data(100, all_test_items, all_test_labels, all_test_pos)

train(model, 50, item_batches, label_batches, pos_batches, test_item_batches, test_label_batches, test_pos_batches, 0.0002)
# 保存模型


torch.save(model, './save/model.pkl')








