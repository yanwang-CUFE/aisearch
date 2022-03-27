import time
from pathlib import Path
import torch
from torch import nn, optim
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
from utils import json_to_dict, write_txt_result, get_data2, drawScore, drawSocre2, drawScore3
import numpy as np
import random
import json
import torch.nn.functional as F

# 设置随机种子
seed = 172
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')

# all_train_query, all_train_document, all_valid_query, all_valid_document = get_all_train_valid_data(all_query, all_document, 2500, 128)
# 直接跑出向量算出相似度

bert_model = 'clue/albert_chinese_tiny'

tokenizer = BertTokenizer.from_pretrained(bert_model)

max_length = 512


# 设计类继承  nn.Module
class BertVectorModel(nn.Module):
    def __init__(self):
        super(BertVectorModel, self).__init__()
        self.bert = AlbertModel.from_pretrained(bert_model)

    def forward(self, ids, mask):
        _, pooled_output = self.bert(
            input_ids=ids,
            attention_mask=mask,
            return_dict=False
        )
        return pooled_output


mymodel = BertVectorModel()


# 获取gpu和cpu的设备信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), "GPUs!")
    # gpu_ids = [1, 0]
    mymodel = nn.DataParallel(mymodel)
mymodel.to(device)


def get_vector(list_, tokenizer, model, max_len):
    model.eval()
    matrix = torch.zeros(len(list_), 312).to(device)
    with torch.no_grad():
        for index, i in enumerate(list_):
            encoding = tokenizer(
                i,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            input_id_tmp = encoding['input_ids'].to(device)
            attention_mask_tmp = encoding['attention_mask'].to(device)
            vector_tmp = model(input_id_tmp, attention_mask_tmp)
            matrix[index] = vector_tmp.squeeze()
    print(matrix.size())
    return matrix


# 将所有的向量和query, document 都保存下来
def get_similar_matrix_k(query_list, document_list):
    length = query_list.size()[0]
    simi = F.cosine_similarity(query_list.unsqueeze(1), document_list.unsqueeze(0), dim=2)
    return simi


def get_top_k_current(similar_matrix, k):
    """
    :param similar_matrix: tensor
    :param k: top k
    :return: current rate
    """
    _, top_k_matrix = torch.topk(similar_matrix, k, 1)
    print(top_k_matrix)
    top_k_matrix = top_k_matrix.cpu().numpy()
    current = 0
    row = top_k_matrix.shape[0]
    print(row)
    column = top_k_matrix.shape[1]
    print(column)
    for i in range(row):
        for j in range(column):
            if i == top_k_matrix[i][j]:
                current += 1
                continue
    return current / row



file_name = "./society1.json"
all_query, all_document = get_data2(file_name)
# all_query = all_query[:20]
# all_document = all_document[:20]
query_vector = get_vector(all_query, tokenizer, mymodel, 128)
document_vector = get_vector(all_document, tokenizer, mymodel, 512)

similar_matrix = get_similar_matrix_k(query_vector, document_vector)
# print(similar_matrix)
print("end")
current_rate = get_top_k_current(similar_matrix, 10)
print(current_rate)


