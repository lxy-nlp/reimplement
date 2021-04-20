# !/home/lxy/anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/21 下午7:23
# @Author  : lxy
# @FileName: test.py
# @Software: PyCharm

import numpy as np
import torch

from mytask.SemTask8.utils.model import SimpleModel
from mytask.SemTask8.utils.utils import evaluate_instance_based
from mytask.SemTask8.utils.vocab import Vocab, read_labels, read_datas, load_data


def test(model,item_batches, label_batches, pos_batches):
    model.eval()
    truth_labels = []
    pre_labels = []
    count = 0
    for (batchx, batchy), pos in zip(zip(item_batches, label_batches), pos_batches):
        sentence_ids = list(np.array(batchx)[:, 0])
        sub_pos = list(np.array(batchx)[:, 1])
        obj_pos = list(np.array(batchx)[:, 2])
        mask_matrix = list(np.array(batchx)[:, 3])
        pos_emb = list(np.array(pos))
        result_class = model(sentence_ids, pos_emb, sub_pos, obj_pos, mask_matrix)
        batchy = torch.tensor(batchy, dtype=torch.int64)
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
    print('测试集预测精准率是{},召回率是{},f1是{}'.format(p,r,f1))


vocab = Vocab('../datas/pre_embeddings/glove/glove.6B.200d.txt', 200, True)
labels, labels_index = read_labels('../datas/SemEval2010-Task8/test/test_result.txt')
all_items, labels, all_pos, _ = read_datas('../datas/SemEval2010-Task8/test/test.txt', vocab.word2id, labels, labels_index)
item_batches, label_batches, pos_batches = load_data(50, all_items, labels, all_pos)
model = torch.load('./save')
# test(model, item_batches, label_batches, pos_batches)
