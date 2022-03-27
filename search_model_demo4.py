import time
from pathlib import Path
import torch
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from utils import json_to_dict, write_txt_result
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# 获取数据
def get_data(content_dict):
    """
    获取模型需要的数据
    :param content_dict: 每个类型的数据字典
    :return: query, document
    """
    dict_all = json_to_dict(content_dict)
    query = []
    document = []
    for id_i in dict_all:
        dict_i = dict_all[id_i]
        title = dict_i['title']
        desc = dict_i['desc']
        answer = dict_i['answer']
        if desc == "":
            query.append(title)
        else:
            query.append(desc)
        document.append(answer)

    # length = len(query)
    # labels = np.eye(length)

    return query, document


def torch_show_all_params(model, rank=0):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    if rank == 0:
        print("Total param num：" + str(k))


def get_train_valid_data(query, document):
    """
    获取训练集，验证集
    :param query:
    :param document:
    :return:
    """
    train_query = []
    train_document = []
    valid_query = []
    valid_document = []
    for id_i, query_i in enumerate(query):
        if id_i % 5 != 0:
            train_query.append(query[id_i])
            train_document.append(document[id_i])
        else:
            valid_query.append(query[id_i])
            valid_document.append(document[id_i])
    train_label = np.eye(len(train_query))
    valid_label = np.eye(len(valid_query))
    print("训练集:  " + str(len(train_query)) + "条")
    print("验证集   " + str(len(valid_query)) + "条")
    return train_query, train_document, train_label, valid_query, valid_document, valid_label


def get_all_train_valid_data(query, document, t_size, v_size):
    """
    获取总训练集，验证集
    :param query:
    :param document:
    :return:
    """
    train_query = query[:t_size]
    train_document = document[:t_size]
    valid_query = query[:v_size]
    valid_document = document[:v_size]

    train_label = np.eye(len(train_query))
    valid_label = np.eye(len(valid_query))
    print("训练集总数:  " + str(len(train_query)) + "条")
    print("验证集   " + str(len(valid_query)) + "条")
    return train_query, train_document, valid_query, valid_document


print("train file load start")
file_name = "./2778category社会民生-公务办理.json"
all_query, all_document = get_data(file_name)
all_train_query, all_train_document, all_valid_query, all_valid_document = get_all_train_valid_data(all_query, all_document, 2500, 128)
all_valid_label = np.eye(len(all_valid_query))

# test_query, test_document, test_label = get_test_data(all_query, all_document, 16)
# train_query, train_document, train_label, valid_query, valid_document, valid_label = get_train_valid_data(all_query, all_document, 2500, 128)
# train_query, train_document, train_label, valid_query, valid_document, valid_label = get_train_valid_data(all_query,
#                                                                                                           all_document,
#                                                                                                           2500, 128)

train_size = 256
valid_size = len(all_valid_query)

# 选择预训练模型  根据bert model 初始化 tokenizer
# 如果token要封装到自定义model类中的话，则需要指定max_len
# bert_model = './bert_base_chinese'
# bert_model = 'voidful/albert_chinese_base'
bert_model = 'clue/albert_chinese_tiny'

tokenizer = BertTokenizer.from_pretrained(bert_model)

# 初始化最大长度
max_length = 256


# 构建自己的 dataset 类， 用于加载数据
class DataToDataset(Dataset):
    def __init__(self, querys, documents, labels, tokenizer, max_len):
        self.querys = querys
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.querys)

    def __getitem__(self, index):
        """
              item 为数据索引，迭代取第item条数据
        """
        text0 = str(self.querys[index])
        print(text0)
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for i_index, i in enumerate(self.documents):
            # print(index)
            # print(i_index)
            text1 = str(self.documents[i_index])
            label = self.labels[index][i_index]
            encoding = self.tokenizer.encode_plus(
                text0,
                text1,
                max_length=self.max_len,
                padding='max_length',
                truncation='longest_first',
                return_tensors='pt',
            )
            input_ids_list.append(encoding['input_ids'].flatten())
            attention_mask_list.append(encoding['attention_mask'].flatten())
            labels_list.append(label)
        return {
            'query': text0,
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list
        }


val_dataset = DataToDataset(all_valid_query, all_valid_document, all_valid_label, tokenizer, max_length)

# 构建 dataloader 类
BATCH_SIZE = 1

import random


def get_train_data_loader(query, document, tokenizer, sample_num, max_length):

    sample_list = [i for i in range(len(query))]  # [0, 1, 2, 3, 4, 5, 6, 7]
    sample_list = random.sample(sample_list, sample_num)  # 随机选取出了 [3, 4, 2, 0]
    train_query = [query[i] for i in sample_list]  # ['d', 'e', 'c', 'a']
    train_document = [document[i] for i in sample_list]  # [3, 4, 2, 0]
    train_label = np.eye(len(train_query))
    train_dataset = DataToDataset(train_query, train_document, train_label, tokenizer, max_length)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1)
    return train_loader

train_loader = get_train_data_loader(all_train_query, all_train_document, tokenizer, 256, max_length)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

# 设计类继承  nn.Module
class BertAISearchModel(nn.Module):
    def __init__(self):
        super(BertAISearchModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.drop = nn.Dropout(p=0.3)
        self.dense = nn.Linear(self.bert.config.hidden_size, 1)  # 768 input, 1 output

    def forward(self, ids, mask):

        _, pooled_output = self.bert(
            input_ids=ids,
            attention_mask=mask,
            return_dict=False
        )
        output = self.drop(pooled_output)  # dropout
        one_output = self.dense(output)

        return one_output


mymodel = BertAISearchModel()
torch_show_all_params(mymodel)

# 获取gpu和cpu的设备信息
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device=", device)
# if torch.cuda.device_count() > 1:
#     print("Let's use ", torch.cuda.device_count(), "GPUs!")
#     mymodel = nn.DataParallel(mymodel)
mymodel.to(device)


class PairwiseMaxMarginHingeLossFunc(nn.Module):
    def __init__(self, batch_size):
        super(PairwiseMaxMarginHingeLossFunc, self).__init__()
        self.batch_size = batch_size

    def forward(self, out, label, margin):
        out = out.t()
        label = label.t()
        loss = 0
        for i in range(self.batch_size):
            out_row = out[i]
            label_row = label[i]
            matrix_size = out_row.size()[0]
            loss_matrix = torch.zeros(matrix_size, matrix_size)

            for j in range(out_row.size()[0]):
                t1 = out_row[j] - out_row
                t2 = label_row[j] - label_row
                loss_matrix[j] = t1 * t2
            loss_matrix = margin - loss_matrix
            big_loss_matrix = torch.clamp(loss_matrix, 0.0)  # 将小于0的元素变为0。
            loss_i = big_loss_matrix.sum() / (matrix_size * 2)
            loss += loss_i
        return loss

optimizer = Adam(mymodel.parameters(), lr=5e-5)

# 计算准确率
def flat_accuracy(preds, labels):
    pred_max = torch.argmax(preds, 1)
    # print(pred_max)
    labels_max = torch.argmax(labels, 1)
    # print(labels_max)
    return torch.sum(pred_max == labels_max)

def flat_accuracy2(preds, labels, txt):
    pred_max = torch.argmax(preds, 1)
    # print(pred_max)
    labels_max = torch.argmax(labels, 1)
    # print(labels_max)
    log_str = str(pred_max) + " " + str(labels_max)
    write_txt_result(txt, log_str)
    return torch.sum(pred_max == labels_max)

loss_func_distance = 1

epochs = 20
for epoch in range(epochs):
    print("epoch   " + str(epoch) + "!!!!")
    print("****************")
    losses = []
    correct_predictions = 0
    mymodel.train()
    train_loader = get_train_data_loader(all_train_query, all_train_document, tokenizer, 256, max_length)
    for data in tqdm(train_loader):
        input_ids_list = data["input_ids"]
        attention_mask_list = data["attention_mask"]
        labels_list = data["labels"]
        out_list = []

        out_matrix = torch.zeros([len(input_ids_list), BATCH_SIZE])

        label_matrix = torch.zeros([len(input_ids_list), BATCH_SIZE])
        # 优化器置零
        optimizer.zero_grad()
        for index, i in enumerate(input_ids_list):
            # print(index)
            input_ids = input_ids_list[index].to(device)
            attention_mask = attention_mask_list[index].to(device)
            labels = labels_list[index].to(device)

            # 得到模型的结果
            out = mymodel(input_ids, attention_mask)
            out_list.append(out)

            out_matrix[index] = out.t()
            label_matrix[index] = labels

        lossFunc = PairwiseMaxMarginHingeLossFunc(BATCH_SIZE).to(device)
        loss = lossFunc(out_matrix, label_matrix, 0)

        # 计算准确率
        current_num = flat_accuracy(out_matrix.t(), label_matrix.t())
        losses.append(loss)
        correct_predictions += current_num

        # 误差反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()
    print("准确数目")
    print(correct_predictions)
    # ("准确率")
    accuracy = correct_predictions.double() / train_size
    # ("平均损失")
    mean_loss = torch.mean(torch.tensor(losses))

    # 计算acc

    print("train %d/%d epochs Loss:%f, Acc:%f" % (epoch, epochs, mean_loss, accuracy))
    log_str = "train  " + str(epoch) + "   mean_loss  " + str(mean_loss) + "   num   " + str(correct_predictions) + "  accuracy  " + str(accuracy)
    write_txt_result("log3.txt", log_str)

    print("------------")

    print("evaluate...")
    val_loss = 0
    val_acc = 0
    losses2 = []
    correct_predictions2 = 0
    mymodel.eval()


    print(len(val_loader))
    with torch.no_grad():
        valid_txt = "valid" + str(epoch) + ".txt"
        for d in tqdm(val_loader):

            query = d["query"][0]
            write_txt_result(valid_txt, query)
            input_ids_list = d["input_ids"]
            attention_mask_list = d["attention_mask"]
            labels_list = d["labels"]
            out_list = []

            out_matrix = torch.zeros([len(input_ids_list), BATCH_SIZE])

            label_matrix = torch.zeros([len(input_ids_list), BATCH_SIZE])
            # 优化器置零
            optimizer.zero_grad()
            for index, i in enumerate(input_ids_list):
                input_ids = input_ids_list[index].to(device)
                attention_mask = attention_mask_list[index].to(device)
                labels = labels_list[index].to(device)

                # 得到模型的结果
                out = mymodel(input_ids, attention_mask)
                out_list.append(out)

                out_matrix[index] = out.t()
                label_matrix[index] = labels

            lossFunc = PairwiseMaxMarginHingeLossFunc(BATCH_SIZE).to(device)
            loss = lossFunc(out_matrix, label_matrix, 0)
            current_num = flat_accuracy2(out_matrix.t(), label_matrix.t(), valid_txt)
            write_txt_result(valid_txt, " ")
            losses2.append(loss)
            correct_predictions2 += current_num
        print("准确数目")
        print(correct_predictions2)
        # ("准确率")
        accuracy2 = correct_predictions2.double() / valid_size
        # ("平均损失")
        mean_loss2 = torch.mean(torch.tensor(losses2))
        print("evaluate loss:%f, Acc:%f" % (mean_loss2, accuracy2))
        log_str2 = "valid  " + str(epoch) + "   mean_loss  " + str(mean_loss2) + "   num   " + str(correct_predictions2)  + "  accuracy  " + str(accuracy2)
        write_txt_result("log3.txt", log_str2)



