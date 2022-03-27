# from pathlib import Path
#
# from tqdm import tqdm
#
#
# def read_imdb_split(split_dir):
#     split_dir = Path(split_dir)
#     texts = []
#     labels = []
#     for label_dir in ["pos", "neg"]:
#         for text_file in (split_dir / label_dir).iterdir():
#             # print(text_file)
#             texts.append(text_file.read_text(encoding="utf-8"))
#             labels.append(0 if label_dir is "neg" else 1)
#
#     return texts, labels
#
#
# print("train file load start")
# train_texts, train_labels = read_imdb_split('aclImdb/train')
# print("train file load end")
# test_texts, test_labels = read_imdb_split('aclImdb/test')
#
# from sklearn.model_selection import train_test_split
#
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
#
# from transformers import DistilBertTokenizerFast
#
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#
# print("train_encoding start")
# train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)
# print("train_encoding end")
#
# import torch
#
#
# class IMDbDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#
#     def __len__(self):
#         return len(self.labels)
#
#
# print("dataset start")
# train_dataset = IMDbDataset(train_encodings, train_labels)
# val_dataset = IMDbDataset(val_encodings, val_labels)
# test_dataset = IMDbDataset(test_encodings, test_labels)
# print("dataset end")
#
# # from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
# #
# # training_args = TrainingArguments(
# #     output_dir='./results',          # output directory
# #     num_train_epochs=3,              # total number of training epochs
# #     per_device_train_batch_size=16,  # batch size per device during training
# #     per_device_eval_batch_size=64,   # batch size for evaluation
# #     warmup_steps=500,                # number of warmup steps for learning rate scheduler
# #     weight_decay=0.01,               # strength of weight decay
# #     logging_dir='./logs',            # directory for storing logs
# #     logging_steps=10,
# # )
# #
# # model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# #
# # trainer = Trainer(
# #     model=model,                         # the instantiated 🤗 Transformers model to be trained
# #     args=training_args,                  # training arguments, defined above
# #     train_dataset=train_dataset,         # training dataset
# #     eval_dataset=val_dataset             # evaluation dataset
# # )
# #
# # trainer.train()
#
#
# from torch.utils.data import DataLoader
# from transformers import DistilBertForSequenceClassification, AdamW
#
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
# model = DistilBertForSequenceClassification.from_pretrained('./distilbert_base_uncased')
# model.to(device)
# model.train()
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
#
# optim = AdamW(model.parameters(), lr=5e-5)
#
#
# # def train_epoch(amodel, data_loader):
# #     model = amodel.train()
#
# import numpy as np
# def flat_accuracy(preds, labels):
#     pred_flat = np.argmax(preds, axis=1).flatten() # [3, 5, 8, 1, 2, ....]
#     labels_flat = labels.flatten()
#     return np.sum(pred_flat == labels_flat) / len(labels_flat)
#
# def eval(model, val_loader):
#     model.eval()
#     eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
#     for batch in val_loader:
#         batch = tuple(t.to(device) for t in batch)
#         with torch.no_grad():
#             logits = model(batch[0], attention_mask=batch[1])[0]#注意跟下面的参数区别，这个地方model.eval()指定是测试，所以没有添加label
#             logits = logits.detach().cpu().numpy()
#             label_ids = batch[2].cpu().numpy()
#             tmp_eval_accuracy = flat_accuracy(logits, label_ids)
#             eval_accuracy += tmp_eval_accuracy
#             nb_eval_steps += 1
#     print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
#
#
# for epoch in range(3):
#     for i, batch in enumerate(train_loader):
#         model.train()
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()
#         if i % 10 == 0:
#             eval(model, val_loader)
#
# model.eval()
