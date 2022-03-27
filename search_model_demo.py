import time
from pathlib import Path
import torch
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from utils import json_to_dict
import numpy as np


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            # print(text_file)
            texts.append(text_file.read_text(encoding="utf-8"))
            labels.append(0 if label_dir is "neg" else 1)
    return texts, labels


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


# 构建数据集，需要query， document， label

# def get_query_document_data():
#     # train = [["第一个问题", "第一个问题的答案"],
#     #          ["第二个问题", "第二个问题的答案"],
#     #          ["第一个问题", "第十个问题的答案"],
#     #          ["第十个问题", "第一个问题的答案"]]
#     #
#     # train_labels = [1, 1, 0, 0]
#     querys = ["the first question", "the second question", "the third question", "the fourth question", "the fifth question"]
#     documents = ["the first answer", "the second answer", "the third answer", "the fourth answer", "the fifth answer"]
#     labels = [[1, -1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, -1, 1, -1, -1], [-1, -1, -1, 1, -1], [-1, -1, -1, -1, 1]]
#     return querys, documents, labels
#
# # train_texts, train_labels = read_imdb_split('aclImdb/train')
# query_texts, document_texts, train_labels = get_query_document_data()


print("train file load start")
file_name = "./2778category社会民生-公务办理.json"
all_query, all_document = get_data(file_name)

train_query, train_document, train_label, valid_query, valid_document, valid_label = get_train_valid_data(all_query, all_document)
train_size = len(train_query)
valid_size = len(valid_query)

print(train_query[0])
print(train_document[0])
print(train_label[0][0])
print("train file load end")


# 选择预训练模型  根据bert model 初始化 tokenizer
# 如果token要封装到自定义model类中的话，则需要指定max_len
bert_model = './bert_base_chinese'
tokenizer = BertTokenizer.from_pretrained(bert_model)

from torch.utils.data import Dataset, DataLoader, random_split


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
                truncation='only_second',
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


# 封装数据
# datasets = DataToDataset(query_texts, document_texts, train_labels, tokenizer, max_length)
#
#
# train_size = int(len(datasets) * 0.8)
# test_size = len(datasets) - train_size
# print([train_size, test_size])
# # 分配train data  val data
# train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, test_size])

# 封装train  valid  data
train_dataset = DataToDataset(train_query, train_document, train_label, tokenizer, max_length)
val_dataset = DataToDataset(valid_query, valid_document, valid_label, tokenizer, max_length)


# 构建 dataloader 类
BATCH_SIZE = 1
# 这里的num_workers要大于0,是抽取数据的线程数目
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 查看 dataloader 结构
# datalp = next(iter(train_loader))
# print(datalp)
# print("*******")
# print(datalp.keys())

# val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 设计类继承  nn.Module
class BertAISearchModel(nn.Module):
    def __init__(self):
        super(BertAISearchModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.drop = nn.Dropout(p=0.3)
        self.dense = nn.Linear(768, 1)  # 768 input, 1 output

    def forward(self, ids, mask):
        # out, _ = self.bert(input_ids=ids, attention_mask=mask)
        # out = self.dense(out[:, 0, :])
        # return out
        _, pooled_output = self.bert(
            input_ids=ids,
            attention_mask=mask,
            return_dict=False
        )
        output = self.drop(pooled_output)  # dropout
        one_output = self.dense(output)
        # return torch.sigmoid(one_output)
        return one_output


mymodel = BertAISearchModel()

# 获取gpu和cpu的设备信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), "GPUs!")
    mymodel = nn.DataParallel(mymodel)
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
            big_loss_matrix= torch.clamp(loss_matrix, 0.0)  # 将小于0的元素变为0。
            loss_i = big_loss_matrix.sum() / (matrix_size * 2)
            loss += loss_i
        return loss


# pairwise max margin hinge loss function
# def pairwise_max_margin_hinge_loss(pertinence, labels, distance):
#     loss_all = 0
#     length = len(labels)
#     for i, pertinence_i in enumerate(pertinence):
#         for j, pertinence_j in enumerate(pertinence):
#             if i != j:
#                 temp_score = distance - (pertinence_i - pertinence_j) * (labels[i] - labels[j])
#                 loss_all += max(0, temp_score)
#     return loss_all / (length * (length - 1))

# loss_func = nn.CrossEntropyLoss().to(device)
# loss_func = pairwise_max_margin_hinge_loss().to(device)

# loss_func = PairwiseMaxMarginHingeLossFunc(BATCH_SIZE).to(device)

optimizer = Adam(mymodel.parameters(), lr=5e-5)

from sklearn.metrics import accuracy_score
import numpy as np
# 计算准确率
def flat_accuracy(preds, labels):
    pred_max = torch.argmax(preds, 1)
    labels_max = torch.argmax(labels, 1)
    return torch.sum(pred_max == labels_max)


loss_func_distance = 1

epochs = 5
for epoch in range(epochs):
    print("epoch   " + str(epoch) + "!!!!")
    print("****************")
    losses = []
    correct_predictions = 0
    mymodel.train()
    for data in tqdm(val_loader):
        input_ids_list = data["input_ids"]
        attention_mask_list = data["attention_mask"]
        labels_list = data["labels"]
        # print("input_ids")
        # print(input_ids_list)
        # print("")
        # print("attention_mask")
        # print(attention_mask_list)
        # print("")
        # print("label_list")
        # print(labels_list)
        # print("")
        out_list = []

        #
        out_matrix = torch.zeros([len(input_ids_list), BATCH_SIZE])
        # print(out_matrix)
        # print(out_matrix.size())
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

        # 计算误差
    #     _, preds = torch.max(out, dim=1)
    #     print(out_list)
        # for index, tensor_i in out_list:
        #     out_matrix[index] = tensor_i

        # for index, tensor_i in labels_list:
        #     label_matrix[index] = tensor_i
        # print("**")
        # print(out_matrix)
        # print(label_matrix)
        lossFunc = PairwiseMaxMarginHingeLossFunc(BATCH_SIZE).to(device)
        loss = lossFunc(out_matrix, label_matrix, 0)
        # print("loss")
        # print(loss)
        # out_list = torch.squeeze(out_list)
        # print(out_list)
        # loss = loss_func(out_list, labels_list, loss_func_distance)
    #     correct_predictions += torch.sum(preds == labels)
    #     losses.append(loss.item())
    #     # train_loss += loss.item()
        # 计算准确率
        current_num = flat_accuracy(out_matrix, label_matrix)
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
    mean_loss = torch.mean(losses)



    # 计算acc
    # out = out.detach().numpy()
    # labels = labels.detach().numpy()
    # train_acc += flat_accuracy(out, labels)

    print("train %d/%d epochs Loss:%f, Acc:%f" % (epoch, epochs, mean_loss, accuracy))
    print("------------")

    print("evaluate...")
    val_loss = 0
    val_acc = 0
    losses2 = []
    correct_predictions2 = 0
    mymodel.eval()
    # for j, batch in enumerate(val_loader):
    #     # val_input_ids, val_attention_mask, val_labels = [elem.to(device) for elem in batch]
    #     # val_input_ids, val_attention_mask, val_labels
    #     val_input_ids = batch["input_ids"].to(device)
    #     val_attention_mask = batch["attention_mask"].to(device)
    #     # print(input_ids)
    #     # print(attention_mask)
    #     val_labels = batch["labels"].to(device)
    #     with torch.no_grad():
    #         pred = mymodel(val_input_ids, val_attention_mask)
    #         val_loss += loss_func(pred, val_labels)
    #         pred = pred.detach().cpu().numpy()
    #         val_labels = val_labels.detach().cpu().numpy()
    #         val_acc += flat_accuracy(pred, val_labels)
    # print("evaluate loss:%d, Acc:%d" % (val_loss / len(val_loader), val_acc / len(val_loader)))
    print(len(val_loader))
    with torch.no_grad():
        for d in tqdm(val_loader):

            #         attention_mask2 = d["attention_mask"].to(device)
            #         targets2 = d["labels"].to(device)
            #
            #         outputs2 = mymodel(input_ids2, attention_mask2)
            #         _, preds2 = torch.max(outputs2, dim=1)
            #
            #         loss2 = loss_func(outputs2, targets2)
            #
            #         correct_predictions2 += torch.sum(preds2 == targets2)
            #         losses2.append(loss2.item())

            input_ids_list = d["input_ids"]
            attention_mask_list = d["attention_mask"]
            labels_list = d["labels"]
            out_list = []

            #
            out_matrix = torch.zeros([len(input_ids_list), BATCH_SIZE])
            # print(out_matrix)
            # print(out_matrix.size())
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
            current_num = flat_accuracy(out_matrix, label_matrix)
            losses2.append(loss)
            correct_predictions2 += current_num
        print("准确数目")
        print(correct_predictions2)
        # ("准确率")
        accuracy2 = correct_predictions2.double() / valid_size
        # ("平均损失")
        mean_loss2 = torch.mean(losses2)
        print("evaluate loss:%d, Acc:%d" % (mean_loss2, accuracy2))


