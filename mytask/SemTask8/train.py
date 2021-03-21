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

from mytask.SemTask8.utils.model import SimpleModel
from mytask.SemTask8.utils.utils import evaluate_instance_based
from mytask.SemTask8.utils.vocab import Vocab, read_labels, read_datas,load_data


def train(model, epoches, item_batches, label_batches, pos_batches,learning_rate):
    train_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
    criterion = nn.CrossEntropyLoss()
    # print("{},Start Training".format(datetime.now().strftime('02y/%02m%02d %H:%M:%S')))

    for epoch in range(epoches):
        f1 = 0
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
            optimizer.zero_grad()  # 清空梯度
            batchy = torch.tensor(batchy, dtype=torch.int64)
            train_loss = criterion(result_class, batchy)  # label_y
            train_loss.backward()  # 反向传播
            nn.utils.clip_grad_norm(model.parameters(), 0.15) # 梯度削减策略
            optimizer.step() #
            _,predict = torch.max(result_class, dim=1)
            labels = batchy.reshape(-1).tolist()    # label是输入标签的id值
            predict = predict.data.tolist()
            # p_indices = np.array(labels) != 0
            predict = np.array(predict).tolist()
            labels = np.array(labels).tolist()
            epoch_label += labels
            epoch_pre += predict
        prcision, recall, add_f1 = evaluate_instance_based(epoch_pre, epoch_label)
            # f1 += add_f1
        print(prcision, recall, add_f1)



vocab = Vocab('../datas/pre_embeddings/glove/glove.6B.200d.txt', 200, True)
labels, labels_index = read_labels('../datas/SemEval2010-Task8/train/train_result.txt')
all_items, labels,all_pos = read_datas('../datas/SemEval2010-Task8/train/train.txt', vocab.word2id, labels, labels_index)
item_batches, label_batches,pos_batches = load_data(50, all_items, labels,all_pos)
model = SimpleModel(400002, 200, vocab.emdeddings, 4, 200, 2, 10,bidirection=True)
train(model, 10, item_batches, label_batches, pos_batches,0.001)





