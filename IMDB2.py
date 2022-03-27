from pathlib import Path
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


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


print("train file load start")
train_texts, train_labels = read_imdb_split('aclImdb/train')

print(train_texts[0])
print(train_labels[0])
print("train file load end")

# 如果token要封装到自定义model类中的话，则需要指定max_len
bert_model = './bert_base_uncased'
print("token encoding start")
import torch

tokenizer = BertTokenizer.from_pretrained(bert_model)
max_length = 64

# sentences_tokened=tokenizer(train_texts,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
# targets=torch.tensor(train_labels)
# print("token encoding end")


# from torchvision import transforms,datasets
from torch.utils.data import Dataset, DataLoader, random_split


# class DataToDataset(Dataset):
#     def __init__(self, encoding, labels):
#         self.encoding = encoding
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         return self.encoding['input_ids'][index], self.encoding['attention_mask'][index], self.labels[index]


class DataToDataset(Dataset):
    def __init__(self, atexts, alabels, atokenizer, amax_len):
        self.texts = atexts
        self.labels = alabels
        self.tokenizer = atokenizer
        self.max_len = amax_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


print("dataset start")
# 封装数据
datasets = DataToDataset(train_texts, train_labels, tokenizer, max_length)
train_size = int(len(datasets) * 0.8)
test_size = len(datasets) - train_size
print([train_size, test_size])
train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, test_size])
print("dataset end")

print("dataload start")
BATCH_SIZE = 8
# 这里的num_workers要大于0
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# datalp = next(iter(train_loader))
# print(datalp)
# print("*******")
# print(datalp.keys())

val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)  #


print("dataload end")


class BertTextClassficationModel(nn.Module):
    def __init__(self):
        super(BertTextClassficationModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.drop = nn.Dropout(p=0.3)
        self.dense = nn.Linear(768, 2)  # 768 input, 2 output

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
        return self.dense(output)


mymodel = BertTextClassficationModel()

# 获取gpu和cpu的设备信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), "GPUs!")
    mymodel = nn.DataParallel(mymodel)
mymodel.to(device)

loss_func = nn.CrossEntropyLoss().to(device)
optimizer = Adam(mymodel.parameters(), lr=5e-5)


# 自己实现的 pairwise max margin hinge loss
def pairwise_max_margin_hinge_loss(pertinence, labels, distance):
    loss_all = 0
    length = len(labels)
    for i, pertinence_i in enumerate(pertinence):
        for j, pertinence_j in enumerate(pertinence):
            if i != j:
                temp_score = distance - (pertinence_i - pertinence_j) * (labels[i] - labels[j])
                loss_all += max(0, temp_score)
    return loss_all / (length * (length - 1))
        
        
from sklearn.metrics import accuracy_score
import numpy as np


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


epochs = 3
for epoch in range(epochs):
    print("epoch   " + str(epoch) + "!!!!")
    print("****************")
    train_loss = 0.0
    train_acc = 0.0
    losses = []
    correct_predictions = 0
    mymodel.train()
    for data in tqdm(train_loader):
        # input_ids, attention_mask, labels = [elem.to(device) for elem in data]
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        # print(input_ids)
        # print(attention_mask)
        labels = data["labels"].to(device)
        print(input_ids)
        print(len(input_ids))
        for i in input_ids:
            print(tokenizer.decode(i))
        # 优化器置零
        optimizer.zero_grad()
        # 得到模型的结果
        out = mymodel(input_ids, attention_mask)
        print("out   ===")
        print(out)
        # 计算误差
        _, preds = torch.max(out, dim=1)
        print(preds)
        print(labels)
        loss = loss_func(out, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        # train_loss += loss.item()
        # 误差反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()
    print("准确数目")
    print(correct_predictions)
    print("准确率")
    print(correct_predictions.double() / train_size)
    print("平均损失")
    print(np.mean(losses))
    # 计算acc
    # out = out.detach().numpy()
    # labels = labels.detach().numpy()
    # train_acc += flat_accuracy(out, labels)

    # print("train %d/%d epochs Loss:%f, Acc:%f" % (epoch, epochs, train_loss / (i + 1), train_acc / (i + 1)))

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
            input_ids2 = d["input_ids"].to(device)
            attention_mask2 = d["attention_mask"].to(device)
            targets2 = d["labels"].to(device)

            outputs2 = mymodel(input_ids2, attention_mask2)
            _, preds2 = torch.max(outputs2, dim=1)

            loss2 = loss_func(outputs2, targets2)

            correct_predictions2 += torch.sum(preds2 == targets2)
            losses2.append(loss2.item())
        print("准确数目")
        print(correct_predictions2)
        print("准确率")
        print(correct_predictions2.double() / test_size)
        print("平均损失")
        print(np.mean(losses2))
