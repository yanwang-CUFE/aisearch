import time
from pathlib import Path
import torch
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from utils import json_to_dict, write_txt_result, get_data2, drawScore, drawSocre2
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random

# 获取数据


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


def get_all_train_valid_data(query, document, t_size, v_size):
    """
    获取总训练集，验证集
    :param query:
    :param document:
    :return:
    """
    train_query = query[:t_size]
    train_document = document[:t_size]
    rev_size = 0 - v_size
    valid_query = query[rev_size:]
    valid_document = document[rev_size:]
    print("训练集总数:  " + str(len(train_query)) + "条")
    print("验证集   " + str(len(valid_query)) + "条")

    print("train")
    print(train_query[0] + train_document[0])
    print("valid")
    print(valid_query[0] + valid_document[0])
    return train_query, train_document, valid_query, valid_document


print("train file load start")
file_name = "./society1.json"
all_query, all_document = get_data2(file_name)
all_train_query, all_train_document, all_valid_query, all_valid_document = get_all_train_valid_data(all_query, all_document, 2500, 128)

# TRAIN_SIZE = 1024
# VALID_SIZE = 128

TRAIN_SIZE = 1024
VALID_SIZE = 128

SAMPLE_NUM = 5

def get_sample_data(train_query, train_document, sample_num, num):
    """
    构造随机产生的训练集
    :param train_query:
    :param train_document:
    :param sample_num:
    :param num:
    :return:
    """
    sample_list = [i for i in range(len(train_query))]
    query_return = []
    document_return = []
    label_return = []
    for ind in range(num):
          # [0, 1, 2, 3, 4, 5, 6, 7]
        tmp_list = random.sample(sample_list, sample_num)  # 随机选取出了 [3, 4, 2, 0]
        train_query_t = [train_query[i] for i in tmp_list]  # ['d', 'e', 'c', 'a']
        train_document_t = [train_document[i] for i in tmp_list]  # [3, 4, 2, 0]
        train_label_t = np.eye(sample_num)
        query_return.append(train_query_t)
        document_return.append(train_document_t)
        label_return.append(train_label_t)
    return query_return, document_return, label_return


all_train_query, all_train_document, all_train_label = get_sample_data(all_train_query, all_train_document, SAMPLE_NUM, TRAIN_SIZE)
all_valid_query, all_valid_document, all_valid_label = get_sample_data(all_valid_query, all_valid_document, SAMPLE_NUM, VALID_SIZE)
# 选择预训练模型  根据bert model 初始化 tokenizer
# 如果token要封装到自定义model类中的话，则需要指定max_len
# bert_model = './bert_base_chinese'
# bert_model = 'voidful/albert_chinese_base'
bert_model = 'clue/albert_chinese_tiny'

tokenizer = BertTokenizer.from_pretrained(bert_model)

# 初始化最大长度
max_length = 512


# 构建自己的 dataset 类， 用于加载数据
class DataToDataset(Dataset):
    def __init__(self, querys, documents, labels, tokenizer, max_length):
        self.querys = querys
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.querys)

    def get_luan(self, query, document, label):
        sample_list = [i for i in range(len(query))]
        random.shuffle(sample_list)
        query_return = []
        document_return = []
        label_return = []
        for i in sample_list:
            query_return.append(query[i])
            document_return.append(document[i])
            label_return.append(label[i])
        return query_return, document_return, label_return


    def __getitem__(self, index):
        """
              item 为数据索引，迭代取第item条数据
        """
        queryAll = self.querys[index]
        documentAll = self.documents[index]
        labelAll = self.labels[index]

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for i_index, i in enumerate(queryAll):
            text0 = str(i)
            input_tmp = []
            atten_tmp = []
            label_tmp = []
            for j_index, j in enumerate(documentAll):

                text1 = str(j)
                label = labelAll[i_index][j_index]
                encoding = self.tokenizer.encode_plus(
                    text0,
                    text1,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation='longest_first',
                    return_tensors='pt',
                )
                input_tmp.append(encoding['input_ids'].flatten())
                atten_tmp.append(encoding['attention_mask'].flatten())
                label_tmp.append(label)
            input2, atten2, label2 = self.get_luan(input_tmp, atten_tmp, label_tmp)
            input_ids_list.append(input2)
            attention_mask_list.append(atten2)
            labels_list.append(label2)
        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list
        }

# 构建自己的 train dataset 类， 用于加载数据


# train_dataset = DataToDataset(all_valid_query, all_valid_document, all_valid_label, tokenizer, max_length)
# valid_dataset = DataToDataset(all_train_query, all_train_document, all_train_label, tokenizer, max_length)

valid_dataset = DataToDataset(all_valid_query, all_valid_document, all_valid_label, tokenizer, max_length)
train_dataset = DataToDataset(all_train_query, all_train_document, all_train_label, tokenizer, max_length)
# 构建 dataloader 类
Train_BATCH_SIZE = 8
Valid_BATCH_SIZE = 8

# train_loader = get_train_data_loader(all_train_query, all_train_document, tokenizer, TRAIN_SIZE, max_length)

train_loader = DataLoader(dataset=train_dataset, batch_size=Train_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=valid_dataset, batch_size=Valid_BATCH_SIZE, shuffle=True)


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

        return torch.sigmoid(one_output)


mymodel = BertAISearchModel()
torch_show_all_params(mymodel)

# 获取gpu和cpu的设备信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), "GPUs!")
    mymodel = nn.DataParallel(mymodel)
mymodel.to(device)


class PairwiseMaxMarginHingeLossFunc(nn.Module):
    def __init__(self):
        super(PairwiseMaxMarginHingeLossFunc, self).__init__()

    def flat_accuracy(self, preds, labels):
        pred_max = torch.argmax(preds, 1)
        # print(pred_max)
        labels_max = torch.argmax(labels, 1)
        # print(labels_max)
        return torch.sum(pred_max == labels_max)

    def forward(self, out_matrix, label_matrix, margin, sample_num):
        out_list = torch.split(out_matrix, 1, dim=2)
        label_list = torch.split(label_matrix, 1, dim=2)
        loss = 0
        correction_num = 0
        for index in range(len(out_list)):
            out = torch.squeeze(out_list[index])
            label = torch.squeeze(label_list[index])
            correction_num += self.flat_accuracy(out, label)
            for i in range(sample_num):
                out_row = out[i]
                label_row = label[i]
                matrix_size = sample_num
                loss_matrix = torch.zeros(matrix_size, matrix_size)
                for j in range(sample_num):
                    t1 = out_row[j] - out_row
                    t2 = label_row[j] - label_row
                    loss_matrix[j] = t1 * t2
                loss_matrix = margin - loss_matrix
                big_loss_matrix = torch.clamp(loss_matrix, 0.0)  # 将小于0的元素变为0。
                loss_i = big_loss_matrix.sum()
                loss += loss_i
        return loss, correction_num


optimizer = Adam(mymodel.parameters(), lr=5e-5)

# 计算准确率
# def flat_accuracy(preds, labels):
#     pred_max = torch.argmax(preds, 1)
#     # print(pred_max)
#     labels_max = torch.argmax(labels, 1)
#     # print(labels_max)
#     return torch.sum(pred_max == labels_max)
#
#
# def flat_accuracy2(preds, labels, txt):
#     pred_max = torch.argmax(preds, 1)
#     # print(pred_max)
#     labels_max = torch.argmax(labels, 1)
#     # print(labels_max)
#     # TOP K
#     _, P = torch.topk(preds, 10, 1)
#     log_str = str(pred_max) + " " + str(labels_max) + "\n"
#     write_txt_result(txt, log_str)
#     write_txt_result(txt, str(P))
#     write_txt_result(txt, "-----------")
#     return torch.sum(pred_max == labels_max)



# def draw_all(x, y, name):
#     x0list = []
#     x1list = []
#     y0list = []
#     y1list = []
#     for index, i in y:
#         if i == 1:



txt_path = "log22txt"

# def draw_all(x, y, name):
#     for i in range(len(x)):



epochs = 20
lossFunction = PairwiseMaxMarginHingeLossFunc().to(device)
for epoch in range(epochs):

    print("epoch   " + str(epoch) + "!!!!")
    print("****************")
    losses = []
    out_ret_list = []
    label_ret_list = []
    correct_predictions = 0
    mymodel.train()
    for data in tqdm(train_loader):
        input_ids_list = data["input_ids"]
        attention_mask_list = data["attention_mask"]
        labels_list = data["labels"]
        out_matrix = torch.zeros([SAMPLE_NUM, SAMPLE_NUM, Train_BATCH_SIZE])

        label_matrix = torch.zeros([SAMPLE_NUM, SAMPLE_NUM, Train_BATCH_SIZE])
        # 优化器置零
        optimizer.zero_grad()
        for index, i in enumerate(input_ids_list):
            input_ids_ = input_ids_list[index]
            attention_masks_ = attention_mask_list[index]
            labels_ = labels_list[index]
            for index_j, j in enumerate(input_ids_):
                out = mymodel(j, attention_mask_list[index][index_j])

                out_matrix[index][index_j] = out.t()
                label_matrix[index][index_j] = labels_list[index][index_j].t()
            # 得到模型的结果
        # print("train--")
        # print(out_matrix)
        # print(label_matrix)
        out_matrix_np = out_matrix.clone().detach().numpy()
        label_matrix_np = label_matrix.clone().detach().numpy()
        for x, y in np.nditer([out_matrix_np, label_matrix_np]):
            out_ret_list.append(x)
            label_ret_list.append(y)
        loss, current_num = lossFunction(out_matrix, label_matrix, 0, SAMPLE_NUM)
        # 计算准确率
        losses.append(loss)
        correct_predictions += current_num

        # 误差反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()
    draw_name = "train " + str(epoch)
    drawScore(out_ret_list, label_ret_list, draw_name)
    print(len(out_ret_list))
    index_list = [i for i in range(len(out_ret_list))]
    drawSocre2(index_list, out_ret_list, draw_name + " index", "index", -1, len(out_ret_list) + 1, "socre", 0, 1)
    print("准确数目")
    print(correct_predictions)
    # ("准确率")
    accuracy = correct_predictions.double() / (TRAIN_SIZE * SAMPLE_NUM)
    # ("平均损失")

    mean_loss = torch.mean(torch.tensor(losses))
    # 计算acc
    print("train %d/%d epochs Loss:%f, Acc:%f" % (epoch, epochs, mean_loss, accuracy))
    log_str = "train  " + str(epoch) + "   mean_loss  " + str(mean_loss) + "   num   " + str(
        correct_predictions) + "  accuracy  " + str(accuracy)
    write_txt_result(txt_path, log_str)

    print("------------")
    print("evaluate...")
    val_loss = 0
    val_acc = 0

    losses2 = []
    out_ret_list2 = []
    label_ret_list2 = []
    correct_predictions2 = 0
    mymodel.eval()

    with torch.no_grad():
        for data2 in tqdm(val_loader):
            input_ids_list2 = data2["input_ids"]
            attention_mask_list2 = data2["attention_mask"]
            labels_list2 = data2["labels"]
            out_matrix2 = torch.zeros([SAMPLE_NUM, SAMPLE_NUM, Valid_BATCH_SIZE])

            label_matrix2 = torch.zeros([SAMPLE_NUM, SAMPLE_NUM, Valid_BATCH_SIZE])
            # 优化器置零
            # optimizer.zero_grad()
            for index, i in enumerate(input_ids_list2):
                input_ids_ = input_ids_list2[index]
                attention_masks_ = attention_mask_list2[index]
                labels_ = labels_list2[index]
                for index_j, j in enumerate(input_ids_):
                    out = mymodel(j, attention_mask_list2[index][index_j])
                    out_matrix2[index][index_j] = out.t()
                    label_matrix2[index][index_j] = labels_list2[index][index_j].t()
                # 得到模型的结果
            # print("val--")
            # print(out_matrix2)
            # print(label_matrix2)
            out_matrix2_np = out_matrix2.clone().detach().numpy()
            label_matrix2_np = label_matrix2.clone().detach().numpy()
            for x, y in np.nditer([out_matrix2_np, label_matrix2_np]):
                out_ret_list2.append(x)
                label_ret_list2.append(y)
            loss2, current_num2 = lossFunction(out_matrix2, label_matrix2, 0, SAMPLE_NUM)
            # 计算准确率
            losses2.append(loss2)
            correct_predictions2 += current_num2

    draw_name2 = "valid " + str(epoch)
    drawScore(out_ret_list2, label_ret_list2, draw_name2)
    index_list2 = [i for i in range(len(out_ret_list2))]
    drawSocre2(index_list2, out_ret_list2, draw_name2 + " index", "index", -1, len(out_ret_list2) + 1, "socre", 0, 1)
    print("准确数目")
    print(correct_predictions2)
    # ("准确率")
    accuracy2 = correct_predictions2.double() / (VALID_SIZE * SAMPLE_NUM)
    # ("平均损失")
    mean_loss2 = torch.mean(torch.tensor(losses2))
    print("evaluate loss:%f, Acc:%f" % (mean_loss2, accuracy2))
    log_str2 = "valid  " + str(epoch) + "   mean_loss  " + str(mean_loss2) + "   num   " + str(
        correct_predictions2) + "  accuracy  " + str(accuracy2)
    write_txt_result(txt_path, log_str2)



