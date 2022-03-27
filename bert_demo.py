import random
import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# # 模型的名称
# model_name = "bert-base-uncased"
# # 读取模型对应的tokenizer
# tokenizer = BertTokenizer.from_pretrained(model_name)
# # 载入模型
# model = BertModel.from_pretrained(model_name)
#
# # 测试输入文本
# input_text = "Here is some text to encode"
#
# # 通过tokenizer把文本变成 token_id
# input_ids = tokenizer.encode(input_text, add_special_tokens=True)
# # input_ids:  [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
#
# # 将python数组转化为torch张量
# input_ids = torch.tensor([input_ids])
# # input_ids:  tensor([[  101,  2182,  2003,  2070,  3793,  2000,  4372, 16044,   102]])
#
# # 获得BERT模型最后一个隐层结果
# with torch.no_grad():
#     last_hidden_states = model(input_ids)[0]  # Model outputs are now tuples
# """
# tensor([[[-0.0549,  0.1053, -0.1065,  ..., -0.3550,  0.0686,  0.6506],
#          [-0.5759, -0.3650, -0.1383,  ..., -0.6782,  0.2092, -0.1639],
#          [-0.1641, -0.5597,  0.0150,  ..., -0.1603, -0.1346,  0.6216],
#          ...,
#          [ 0.2448,  0.1254,  0.1587,  ..., -0.2749, -0.1163,  0.8809],
#          [ 0.0481,  0.4950, -0.2827,  ..., -0.6097, -0.1212,  0.2527],
#          [ 0.9046,  0.2137, -0.5897,  ...,  0.3040, -0.6172, -0.1950]]])
# """

query_answer_list = [["第一个问题", "第一个问题的答案"],
                  ["第二个问题", "第二个问题的答案"],
                  ["第一个问题", "第十个问题的答案"],
                  ["第十个问题", "第一个问题的答案"]]

labels = [1, 1, 0, 0]

model_name = 'bert-base-chinese'

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True

device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained(model_name)


def encode_all_texts(text_list):
    all_input_ids = []
    for text_i in text_list:
        input_ids = tokenizer.encode(
            text_i[0], text_i[1],
            add_special_tokens=True,
            max_length=300,
            pad_to_max_length=True,
            return_tensors="pt"

        )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


all_input_ids = encode_all_texts(query_answer_list)
labels = torch.tensor(labels)


def get_cls_encoding(query_answer_list):

    query_answer_result = model(query_answer_list)
    print(query_answer_result)
    # query_result 是一个tuple
    print(query_answer_result[0].size())
    # torch.Size([2, 7, 768])
    print(query_answer_result[0][:, :1, :].size())
    # torch.Size([2, 1, 768])
    print(query_answer_result[0][:, :1, :].squeeze(1).size())
    # torch.Size([2, 768])
    return query_answer_result[0][:, :1, :].squeeze(1)


def get_matching_score(query_cls, document_cls):
    """
    计算单个query和document的相关性
    :param query_cls: query经过模型得出的cls的编码
    :param document_cls: document经过模型得出的cls的编码
    :return: matching_score
    """

    similarity = torch.cosine_similarity(query_cls, document_cls, dim=0)
    return similarity


def get_matrix_score(query_matrix, document_matrix):
    """
    获取两个矩阵的相似度
    :param query_matrix:
    :param document_matrix:
    :return:
    """
    query_matrix_nor = torch.nn.functional.normalize(query_matrix, p=2, dim=1)
    print(query_matrix_nor)
    document_matrix_nor = torch.nn.functional.normalize(document_matrix, p=2, dim=1)
    print(document_matrix_nor.T)
    print(torch.mm(query_matrix_nor, document_matrix_nor.T))



epochs = 3
batch_size = 100

dataset = TensorDataset(all_input_ids, labels)

train_size = int(0.90 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


model = BertModel.from_pretrained(model_name)
model.cuda()


optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        loss,


