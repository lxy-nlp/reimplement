import torch
import torch.nn as nn
import os
from transformers import BertTokenizer,BertModel,BertForMaskedLM,BertConfig
import math
import torch.nn.functional as F

# 这里的参数设置有两种
# 1. 模型名称
# 2. 预训练的模型路径
# 此处选择第二种

# 　加载原始模型
# model_path = './pretrained_model/uncased_bert'
model_path = '/home/lxy/下载/NLP/BERT/test/chinese_L-12_H-768_A-12'
vocab = 'vocab.txt'
config_path = 'bert_config.json'

config = BertConfig.from_json_file(os.path.join(model_path, config_path))
tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, vocab))
bert_model = BertModel.from_pretrained(model_path, config=config)




# 修改输入句子的格式
# [UNK] 100
# [CLS] 101
# [SEP] 102
# [MASK]103
sentenceA = '等潮水退去，就知道谁没穿裤子'
# 构造bert的输入
text_dict = tokenizer.encode_plus(sentenceA, add_special_tokens=True, return_attention_mask=True)

input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)

token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
# attention_mask 0 是被 遮盖  1 未遮盖
attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)

res = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


# bert的简单使用
class CustomModel(nn.Module):
    def __init__(self,bert_path, n_other_features, n_hidden):
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.require_grad = False  # 固定参数 不进行反向传播
        self.output = nn.Sequential(nn.Dropout(0.2),
                                    nn.Linear(768 + n_other_features, n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, 1)
                                    )
    def forward(self,seq,features):
        _,pooled = self.bert(seq,output_all_encoded_layers=False)
        concat = torch.cat([pooled,features], dim=1)
        logits = self.output(concat)
        return logits

torch.cuda.is_available()

class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, head_cnt, h_dim):
        super(Multi_Head_Self_Attention, self).__init__()
        self.m = head_cnt
        self.hidden_dim = int(h_dim / self.m)
        self.q_head = nn.ModuleList()
        self.k_head = nn.ModuleList()
        self.v_head = nn.ModuleList()
        for i in range(self.m):
            self.q_head.append(nn.Linear(h_dim,self.hidden_dim))
            self.k_head.append(nn.Linear(h_dim,self.hidden_dim))
            self.v_head.append(nn.Linear(h_dim,self.hidden_dim))
        self.w = nn.Linear(h_dim, h_dim)
        self.w1 = nn.Linear(h_dim, h_dim)
        self.w2 = nn.Linear(h_dim, h_dim)

    def forward(self,Q, K, V):
        att = torch.bmm(self.q_head[0](Q), self.k_head[0](K), self.v_head[0](V))
        att /= math.sqrt(self.hidden_dim)
        att = F.softmax(att,dim=-1)
        sent = torch.bmm(att, self.v_head[0](V))
        for i in range(1,self.m):
            att = torch.bmm(self.q_head[i][Q],self.k_head[i](K))
            att /= math.sqrt(self.hidden_dim)
            att = F.softmax(att, dim=-1)
            cur_sent = torch.bmm(att, self.v_head[i](V))
            sent = torch.cat((sent,cur_sent),-1)
        sent = self.w(sent)
        sent = nn.LayerNorm(sent.size()[1:],elementwise_affine=False)(sent+Q)
        # nn.ReLU() 是封装好的类,只能在nn.ModuleList中使用  F.relu是函数
        lin_sent = self.w2(nn.ReLU()(self.w1(sent)))
        # elementwise_affine
        # 如果设为False，则LayerNorm层不含有任何可学习参数。
        # 如果设为True（默认是True）则会包含可学习参数weight和bias，用于仿射变换，即对输入数据归一化到均值0方差1后，乘以weight，即bias。
        # nn.** 这样的类型其实是 类 需要实例化后使用
        sent = nn.LayerNorm(sent.size()[1:], elementwise_affine=False)(sent+lin_sent)
        return sent

drop_rate = 0.3
class GCN(nn.Module):
    def __init__(self, num_layers,in_dim, out_dim):
        super(GCN, self).__init__()
        self.drop_rate = drop_rate
        self.gcn_num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        for i in range(self.gcn_num_layers):
            self.gcn_layers.append(nn.Linear(in_dim, out_dim))
        self.W = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, gcn_input, adj):
        att_scores = torch.bmm(self.W(gcn_input), gcn_input.transpose(1, 2))
        exp_att_scores = torch.exp(att_scores)
        combined_att = adj * exp_att_scores
        denom = torch.sum(combined_att,dim=-1) + 1
        norm_att = combined_att /denom.unsqueeze(2)
        for i in range(self.gcn_num_layers):
            Ax = torch.bmm(norm_att, gcn_input)
            AxW = self.gcn_layers[i](Ax)
            AxW = AxW + self.gcn_layers[i](gcn_input)
            gAxW = F.relu(AxW)
            gcn_input = self.dropout(gAxW) if i < self.gcn_num_layers -1 else gAxW
        return gcn_input

batch_size = 30
def train_model(model,train_samples,dev_samples,best_model_file):
    train_size = len(train_samples)
    batch_cnt = int(math.ceil(train_size / batch_size))
    move_last_batch = False
    # 如果最后一个batch只剩一个元素 那么就不用最后一个batch
    if len(train_samples) - batch_size *(batch_cnt - 1) == 1:
        move_last_batch = True
        batch_cnt -= 1

    F.max_pool1d()


if __name__ == '__main__':
    s = "I am not sure,this can work"
    tokens = tokenizer.tokenize(s)
    text_dict = tokenizer.encode_plus(sentenceA, add_special_tokens=True, return_attention_mask=True)
    input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
    # model = CustomModel()