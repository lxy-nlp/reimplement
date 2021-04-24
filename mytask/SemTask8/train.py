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


def train(model, epoches, item_batches, label_batches, pos_batches, dev_item_batches, dev_label_batches, dev_pos_batches, lr, device,norel):
    # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, 100) # 余弦退火法优化策略
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    total_loss = 0
    count = 0
    criterion = nn.CrossEntropyLoss().to(device)
    best_f1 = 0
    for epoch in range(epoches):
        epoch_label = []
        epoch_pre = []
        for (batchx, batchy),pos in zip(zip(item_batches, label_batches), pos_batches):
            sentence_ids = torch.tensor(list(np.array(batchx)[:, 0])).to(device)
            sub_pos = torch.tensor(list(np.array(batchx)[:, 1]),dtype=torch.float32).to(device)
            obj_pos = torch.tensor(list(np.array(batchx)[:, 2]),dtype=torch.float32).to(device)
            mask_matrix = torch.tensor(list(np.array(batchx)[:, 3]),dtype=torch.float32).to(device)
            pos_emb = torch.tensor(list(np.array(pos)),dtype=torch.float32).to(device)
            result_class, attn, pos_weight = model(sentence_ids, pos_emb, sub_pos, obj_pos, mask_matrix)
            optimizer.zero_grad()  # 清空梯度
            batchy = torch.tensor(batchy, dtype=torch.int64).to(device)
            train_loss = criterion(result_class, batchy) + 0.2 * torch.norm(attn, p='fro')
            # + 0.1 * torch.norm(pos_weight, p='fro')
            total_loss += train_loss.cpu()
            train_loss.backward()  # 反向传播
            nn.utils.clip_grad_norm(model.parameters(), 0.25)  # 梯度削减策略
            optimizer.step()
            _,predict = torch.max(result_class, dim=1)
            labels = batchy.reshape(-1).tolist()    # label是输入标签的id值
            predict = predict.data.tolist()
            predict = np.array(predict).tolist()
            labels = np.array(labels).tolist()
            epoch_label += labels
            epoch_pre += predict
            count += 1

        p, r, f1 = evaluate_instance_based(epoch_pre, epoch_label, empty_label=norel)
        print('迭代次数{}'.format(epoch+1))
        print('训练集 精准率是{},召回率是{},f1是{},损失是{}'.format(p, r, f1, total_loss/count))
        pre_labels, truth_labels,loss_dev = test(model, dev_item_batches, dev_label_batches, dev_pos_batches,device)
        p_dev, r_dev, f1_dev = evaluate_instance_based(pre_labels, truth_labels, empty_label=norel)
        print('验证集 精准率是{},召回率是{},f1是{},损失是{}'.format(p_dev, r_dev, f1_dev,loss_dev))
        model.train()
        if f1_dev > best_f1:
            torch.save(model, 'model.pkl')


vocab = Vocab('../datas/pre_embeddings/glove/glove.6B.200d.txt', 200, True)
labels, labels_index,label2id = read_labels('../datas/SemEval2010-Task8/train/train_result.txt')

norel = label2id['Other']
all_items, labels, all_pos, items = read_datas('../datas/SemEval2010-Task8/train/train.txt', vocab.word2id, labels, labels_index)
#
types = [1,2,3,6,7]

added_item, added_label,added_pos = data_argument(items, 0, types, vocab.word2id)
all_items += added_item
labels += added_label
all_pos += added_pos

item_batches, label_batches, pos_batches = load_data(100, all_items, labels,all_pos)
dev_item_batches, dev_label_batches, dev_pos_batches = item_batches[:10],label_batches[:10],pos_batches[:10]
# train_item_batches,train_label_batches,train_pos_batches = item_batches[10:], label_batches[10:], pos_batches[10:]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleModel(400002, 200, vocab.emdeddings, 4, 200, 4, 10, 0.5, device, True, True)
model = model.to(device)

test_labels, test_labels_index,_ = read_labels('../datas/SemEval2010-Task8/test/test_result.txt')
all_test_items, all_test_labels, all_test_pos, _ = read_datas('../datas/SemEval2010-Task8/test/test.txt', vocab.word2id, test_labels, test_labels_index)
test_item_batches, test_label_batches, test_pos_batches = load_data(100, all_test_items, all_test_labels, all_test_pos)

train(model, 50, item_batches, label_batches, pos_batches, test_item_batches, test_label_batches, test_pos_batches, 0.0003,device,norel)

# 测试模型
best_model = torch.load('model.pkl')
# p_test, r_test, f1_test,loss_test = test(best_model, test_item_batches, test_label_batches, test_pos_batches,device)
# print('测试集预测精准率是{},召回率是{},f1是{},损失是{}'.format(p_test, r_test, f1_test, loss_test))










