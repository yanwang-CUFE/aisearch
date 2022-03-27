from transformers import AutoConfig,AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup,logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,SequentialSampler,RandomSampler,DataLoader

import pandas as pd

# 导入transformers
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup


# 导入torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# 常用包
import re
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from tqdm import tqdm


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train = [["第一个问题", "第一个问题的答案"],
                  ["第二个问题", "第二个问题的答案"],
                  ["第一个问题", "第十个问题的答案"],
                  ["第十个问题", "第一个问题的答案"]]

train_labels = [1, 1, 0, 0]

PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
PRE_TRAINED_MODEL_NAME = "./bert_base_chinese"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

print(tokenizer)\

sample_txt = "第一个问题的答案"
sample_txt2 = "第一个问题的结果"


tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f'文本为: {sample_txt}')
print(f'分词的列表为: {tokens}')
print(f'词对应的唯一id: {token_ids}')

encoding=tokenizer.encode_plus(
    sample_txt, sample_txt2,
    max_length=18,
    add_special_tokens=True,# [CLS]和[SEP]
    return_token_type_ids=True,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',# Pytorch tensor张量

)
print(encoding.keys())
print(encoding)

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
out = bert_model(encoding['input_ids'], encoding['attention_mask'])
print(out)
print("+++++++++++++++")
print(len(out))
print(type(out))
for i in out:
    print(i)
