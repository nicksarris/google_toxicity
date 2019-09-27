import warnings
warnings.filterwarnings('ignore')

import os
import time
import gc
import sys

import numpy as np
import pandas as pd
import random
import shutil
import pickle

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from fastai.text import *

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from random import shuffle

from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize.treebank import TreebankWordTokenizer

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam

class MyBertClassifier(BertPreTrainedModel):

    def __init__(self, config, num_aux_targets):
        super(MyBertClassifier, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, 1)
        self.linear_aux_out = nn.Linear(config.hidden_size, num_aux_targets)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        h_conc_linear1  = F.relu(self.linear(pooled_output))
        h_conc_linear1 = self.dropout(h_conc_linear1)

        hidden = pooled_output + h_conc_linear1
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out

def seed_everything(seed=1235):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_data(text_data, max_seq_length,tokenizer):

    max_seq_length -= 2
    all_tokens = []
    longer = 0

    for text in tqdm(text_data):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])
        all_tokens.append(one_token)

    return np.array(all_tokens)

def trim_tensors(tsrs):

    max_len = torch.max(torch.sum( (tsrs[0] != 0  ), 1))
    if max_len > 2:
        tsrs = [tsr[:, :max_len] for tsr in tsrs]

    return tsrs

def calculate_toxicity(test_data):

    batch_size = 1
    max_bert_length = 220
    pytorch_conversion = False

    seed_everything(1235)
    tqdm.pandas()

    bert_model_path = "./service/uncased_L-12_H-768_A-12/"
    trained_model_path = "./service/bert_pytorch_model.bin"
    bert_config = BertConfig(bert_model_path + 'bert_config.json')

    if pytorch_conversion == True:
        convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
            bert_model_path + 'bert_model.ckpt',
            bert_model_path + 'bert_config.json',
            bert_model_path + 'pytorch_model.bin')

    base_tokenizer = BertTokenizer.from_pretrained(bert_model_path, cache_dir=None, do_lower_case=True)
    converted_text = convert_data(test_data, max_bert_length, base_tokenizer)
    bert_test_lengths = torch.from_numpy(np.array([len(x) for x in converted_text]))
    bert_test_set = torch.tensor(pad_sequences(converted_text, maxlen=max_bert_length, padding='post'), dtype=torch.long)

    model = MyBertClassifier(bert_config, 6)
    bert_test_dataset = torch.utils.data.TensorDataset(bert_test_set)
    bert_test_loader = torch.utils.data.DataLoader(bert_test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cpu')
    model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    tk2 = tqdm(enumerate(bert_test_loader), total=len(bert_test_loader), leave=False)

    output_preds = []
    for i, (batch) in tk2:

        tsrs = trim_tensors(batch)
        x_batch, = tuple(t.to(device) for t in tsrs)
        y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        y_pred = torch.sigmoid(torch.tensor(y_pred[:, 0].detach().cpu().squeeze().numpy())).numpy().ravel()
        list.append(output_preds, y_pred)

    return output_preds
