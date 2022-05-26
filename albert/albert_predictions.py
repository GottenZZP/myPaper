import numpy as np
import pandas as pd
from torch import nn
import time
import os
import torch
import json
import logging
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, AlbertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch import softmax
from transformers.utils.notebook import format_time
from albert_modeling import ALBertAndTextCnnForSeq
from albert_processFile import InputDataSet, process_text, read_file, TestInput
from d2l import torch as d2l
from sklearn.metrics import classification_report

device = d2l.try_gpu()

# 读取配置文件
with open("./config.json", mode='r') as f:
    config = f.read()
params = json.loads(config)

# 测试集位置
ZH_test = params["chinese"]["ZH_test"]
JP_test = params["japanese"]["JP_test"]
TEST_PATH = JP_test
# 中文模型地址
ZH_model = params["chinese"]["ZH_model"]
JP_model = params["japanese"]["JP_model"]
NOW_MODEL = JP_model
# 自己的模型保存处
MY_MODEL = params["my_model"]["MY_MODEL_PATH"]
# 下标文件地址
JP_idx = params["japanese"]["JP_idx"]
ZH_idx = params["chinese"]["ZH_idx"]
IDX_PATH = JP_idx if NOW_MODEL == JP_model else ZH_idx
# 预测文件保存地址
ZH_preds = params["chinese"]["ZH_preds"]
JP_preds = params["japanese"]["JP_preds"]
PRED_PATH = JP_preds if params["my_model"]["MY_MODEL_TYPE"] == "japanese" else ZH_preds
file_lens = len(os.listdir(PRED_PATH))
final_path = PRED_PATH + params["use_model"] + f"{file_lens}.csv"
# 预测标签处
TEST_LABEL = params["japanese"]["JP_test_label"] if NOW_MODEL == JP_model else params["chinese"]["ZH_test_label"]


def model_prediction(test_iter, model):
    """预测函数"""
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


def my_prediction(model, testing_loader, test_label, info_name, device):
    """Prediction function"""

    final_file = os.path.join("D:\python_code\paper\data\preds", info_name + "-preds.txt")
    labels = pd.read_csv(test_label)["idx"]
    labels = np.array([x for x in labels])
    lst_prediction = []
    lst_true = []
    lst_prob = []
    model.eval()
    print("Evaluate Start!")
    for step, batch in enumerate(testing_loader):
        print(f"The [{step + 1}]/[{len(testing_loader)}]")
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            probs = softmax(outputs.logits, dim=1)
            logits = torch.argmax(probs, dim=1)
            preds = logits.detach().cpu().numpy()

            lst_prediction.append(preds)
            lst_prob.append(probs)
    print("Evaluate End!")

    lst_true = [int(l) for l in labels]
    lst_prediction = [int(i) for l in lst_prediction for i in l]
    lst_prob = [i.to('cpu').numpy() for prob in lst_prob for i in prob]

    return lst_prob, lst_true


def get_result(pred, lst_true):
    """Get final result"""
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(lst_true, pred)
    f1_micro = f1_score(lst_true, pred, average='micro')
    f1_macro = f1_score(lst_true, pred, average='macro')

    return acc, f1_micro, f1_macro


def avg_prediction(k_result, lst_true):
    k_result = np.array(k_result)
    avg_probs = np.sum(k_result, axis=0) / 5
    avg_probs = torch.from_numpy(avg_probs)
    avg_preds = torch.argmax(avg_probs, dim=1)
    acc, f1_micro, f1_macro = get_result(avg_preds, lst_true)
    print(f"\navg: acc: {acc}, f1_micro: {f1_micro}, f1_macro: {f1_macro}")


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
    df.to_csv(final_path, index=True, sep=',')


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


def cache_info(out_file, text):
    """输出日志"""
    print(text)
    with open(out_file, mode="a+") as f:
        f.writelines(text + '\n')


if __name__ == "__main__":
    test = read_file(TEST_PATH)
    test = process_text(test)
    tokenizer = AlbertTokenizer.from_pretrained(NOW_MODEL)
    test_data = TestInput(test, tokenizer, 512)
    test_iter = DataLoader(test_data, batch_size=16)
    model = ALBertAndTextCnnForSeq.from_pretrained(NOW_MODEL)
    corrects = model_prediction(test_iter, model)
    save_file(corrects)

    # acc = acc_rate("D:\\python_code\\paper\\data\\test_label2.csv", "D:\\python_code\\paper\\data\\my_ans.csv")
    # print(acc)
    
    getEvaReport(TEST_LABEL, final_path, params["my_model"]["MY_MODEL_NAME"][:-4] + str(file_lens))
