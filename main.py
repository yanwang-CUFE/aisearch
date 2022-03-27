from bert_demo import *
import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)