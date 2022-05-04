# coding:utf-8
import numpy as np
import pandas as pd
import time
import os
import re
import torch
import warnings
import json
from transformers import AlbertTokenizer
from torch.utils.data import DataLoader
from transformers import logging
from sklearn.model_selection import train_test_split
from sklearn import datasets
from transformers.utils.notebook import format_time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def read_file(data_dir):
    """读取文件"""
    data = pd.read_csv(data_dir)
    # 将缺省数据Nan填充为空字符
    data['text'] = data['text'].fillna('')
    return data


def process_text(data):
    """将数据中的垃圾字符去除"""
    text = list(data['text'])
    for i in range(len(text)):
        text[i] = text[i].strip().replace('XXXX', '')
    data['text'] = text
    return data


class InputDataSet:

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data['text'][index])
        labels = torch.tensor(self.data['label'][index], dtype=torch.long)

        output = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids, token_type_ids, attention_mask = output.values()
        input_ids = input_ids.squeeze(dim=0)
        attention_mask = attention_mask.squeeze(dim=0)
        token_type_ids = token_type_ids.squeeze(dim=0)
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }


class TestInput:

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data['text'][index])

        output = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids, token_type_ids, attention_mask = output.values()
        input_ids = input_ids.squeeze(dim=0)
        attention_mask = attention_mask.squeeze(dim=0)
        token_type_ids = token_type_ids.squeeze(dim=0)
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }
