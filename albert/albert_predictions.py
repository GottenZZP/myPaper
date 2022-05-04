import numpy as np
import pandas as pd
from torch import nn
import time
import os
import torch
import json
import logging
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time
from zmq import device
from albert_modeling import ALBertAndTextCnnForSeq
from albert_processFile import InputDataSet, process_text, read_file, TestInput
from d2l import torch as d2l
from albert_train import cache_info
from sklearn.metrics import classification_report

device = d2l.try_gpu()

# 读取配置文件
with open("./config.json", mode='r') as f:
    config = f.read()
params = json.loads(config)

# 测试集位置
ZH_test = params["chinese"]["ZH_test"]
JP_test = "D:\python_code\paper\data\\test.csv"
TEST_PATH = JP_test
# 中文模型地址
ZH_model = "D:\\python_code\\paper\\models\\bert-base-chinese"
JP_model = "D:\python_code\paper\models\\bert-base-japanese"
NOW_MODEL = JP_model
# 自己的模型保存处
MY_MODEL = "D:\\python_code\\paper\\models\\chinese\\textcnn-model.bin"
MY_MODEL = "D:\python_code\paper\models\japanese\clean_model_textcnn\\13-model.bin"
# 下标文件地址
JP_idx = "D:\python_code\paper\data\idx.csv"
ZH_idx = "D:\python_code\paper\corpus\chinese\chinese_idx_35000.csv"
IDX_PATH = JP_idx
# 预测文件保存地址
ZH_preds = "D:\\python_code\\paper\\corpus\\chinese\\preds\\textcnn-ans.csv"
JP_preds = "D:\python_code\paper\data\preds\\textcnn-ans.csv"
PRED_PATH = JP_preds


def model_prediction(test_iter, model):
    """预测函数"""
    # model.load_state_dict(torch.load("D:\\python_code\\金融评论分类\\cache\\model_stu.bin"))
    checkpoint = torch.load(MY_MODEL)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    corrects = []
    model.eval()
    print("Evaluate Start!")
    for step, batch in enumerate(test_iter):
        print(f"The [{step + 1}]/[{len(test_iter)}]")
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)

            logits = torch.argmax(outputs.logits, dim=1)
            preds = logits.detach().cpu().numpy()
            
            corrects.append(preds)
    print("Evaluate End!")
    return corrects


def save_file(corrects):
    """保存预测文件"""
    total_ans = []
    for batch in corrects:
        for ans in batch:
            total_ans.append(int(ans))

    df1 = pd.read_csv(IDX_PATH)
    idx, labels = df1["idx"], df1["label"]
    index_to_label = dict()
    for l, i in zip(labels, idx):
        index_to_label[i] = l
    
    final_ans = []
    for n in total_ans:
        final_ans.append([n, index_to_label[n]])
    
    df = pd.DataFrame(final_ans, columns=["idx", "label"])
    df.to_csv(PRED_PATH, index=True, sep=',')


def acc_rate(label_path, ans_path):
    """准确率计算"""
    df1 = pd.read_csv(label_path)
    df2 = pd.read_csv(ans_path)

    test_label = df1["idx"]
    ans_label = df2["idx"]
    acc = (test_label == ans_label).mean()
    for i, ta in enumerate(zip(test_label, ans_label)):
        t, a = ta
        if t != a:
            print(i)
    return acc


def getEvaReport(test_label, test_pred, file_name):
    """评价指标"""
    df1 = pd.read_csv(test_label)
    df2 = pd.read_csv(test_pred)
    labels = df1["idx"].to_list()
    preds = df2["idx"].to_list()
    target_names = pd.read_csv(IDX_PATH)["label"].to_list()
    # target_names = [f"class: {x}" for x in target_names]
    res = classification_report(y_true=labels, y_pred=preds, target_names=target_names, digits=4, output_dict=True)
    # print(res)
    df3 = pd.DataFrame(res)
    out = pd.DataFrame(df3.values.T, index=df3.columns, columns=df3.index)
    print(out)
    out.to_csv(f"D:\\python_code\\paper\\report\\{file_name}.csv")


if __name__ == "__main__":
    test = read_file(TEST_PATH)
    test = process_text(test)
    tokenizer = BertTokenizer.from_pretrained(NOW_MODEL)
    test_data = TestInput(test, tokenizer, 512)
    test_iter = DataLoader(test_data, batch_size=16)
    model = BertForSeq.from_pretrained(NOW_MODEL)
    corrects = model_prediction(test_iter, model)
    save_file(corrects)

    # acc = acc_rate("D:\\python_code\\paper\\data\\test_label2.csv", "D:\\python_code\\paper\\data\\my_ans.csv")
    # print(acc)
    
    getEvaReport("D:\python_code\paper\data\\test_label2.csv", PRED_PATH, "textcnn-ans")