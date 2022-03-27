from sklearn.metrics import accuracy_score
import numpy as np
import torch

# def flat_accuracy(preds, labels):
#     pred_flat = np.argmax(preds, axis=1)
#     print(pred_flat)
#     labels_flat = labels
#     print(labels_flat)
#     return accuracy_score(labels_flat, pred_flat)

#
# a = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.2]])
# b = torch.tensor([[0, 0, 1], [0, 1, 0]])
# def flat_accuracy(preds, labels):
#     pred_max = torch.argmax(preds, 1)
#     labels_max = torch.argmax(labels, 1)
#     return torch.sum(pred_max == labels_max)
#
# c = flat_accuracy(a, b)
#
# print(a)
# print(b)
# print(c)

# a= [1,2 ,3,4 ,5 ,6 ,7]
# b = a[:3]
# c = a[4:]
# print(b)
# print(c)
# import random
# data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
# label = [0, 1, 2, 3, 4, 5, 6, 7]
# sample_num = int(0.5 * len(data))  # 假设取50%的数据
#
# sample_list = [i for i in range(len(data))]  # [0, 1, 2, 3, 4, 5, 6, 7]
# sample_list = random.sample(sample_list, sample_num)  # 随机选取出了 [3, 4, 2, 0]
# sample_data = [data[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
# sample_label = [label[i] for i in sample_list]  # [3, 4, 2, 0]
# print(sample_data)
# print(sample_label)
import torch
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from utils import json_to_dict, write_txt_result
from torch.utils.data import Dataset, DataLoader, random_split
# bert_model = 'clue/albert_chinese_tiny'
#
# tokenizer = BertTokenizer.from_pretrained(bert_model)

# label_test = torch.tensor([[[1,2], [3,4],[5,6]],[[11,12], [13,14],[15,16]],[[21,22], [23,24],[25,26]]])
# print(label_test.size())
# a = torch.split(label_test, 1, dim=2)

#
# A = torch.empty(4, 1, 2, 2)
# with open("test1.txt", "a") as f:
#     f.write(str(A))
#
# print(A.device)
#
# print(A)
#
# print(A.cuda().device)
#
# print(A.cuda())
#
# # A.cuda()
# A = A.cuda()
# print(A.device)
# A = A.to('cuda:0')

import torch
import torch.nn.functional as F
a = torch.rand((4, 64))
b = torch.rand((7, 64))
simi = F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)
print(simi.shape)
print(simi)