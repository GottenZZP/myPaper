import numpy as np
import pandas as pd
from torch import nn
import time
import os
import torch
import random
import json
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME
from transformers import AlbertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time
from albert_modeling import ALBertForSeq, ALBertAndTextCnnForSeq
from albert_processFile import InputDataSet, read_file, TestInput
from albert_predictions import my_prediction, avg_prediction
from d2l import torch as d2l
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

devices = d2l.try_all_gpus()

# 读取配置文件
with open("./config.json", mode='r') as f:
    config = f.read()
params = json.loads(config)

# 中文参数
ZH_model = params['chinese']['ZH_model']
ZH_train = params['chinese']['ZH_train']
ZH_val = params['chinese']['ZH_val']
ZH_save = params['chinese']['ZH_save']
# 日本参数
JP_model = params['japanese']['JP_model']
JP_train = params['japanese']['JP_train']
JP_val = params['japanese']['JP_val']
JP_save = params['japanese']['JP_save']
JP_test = params['japanese']['JP_test']
# 当前使用模型
use_model = params["use_model"]
now_model = JP_model
now_train = ZH_train if now_model == ZH_model else JP_train
now_val = ZH_val if now_model == ZH_model else JP_val
now_save = ZH_save if now_model == ZH_model else JP_save
# 全局随机种子
GLOBAL_SEED = params['GLOBAL_SEED']
# 线程数
GLOBAL_WORKER_ID = 4
num_workers = 4
# tensorboard路径
log_path = params["tensorboard"]
# 学习率
LR = params['LR']
# 学习率预热
warm_up_ratio = params["warm_up_ratio"]
# 句子最大长度
SEQ_MAX_LEN = params["SEQ_MAX_LEN"]

writer = SummaryWriter(log_path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def cache_info(out_file, text):
    """输出日志"""
    print(text)
    with open(out_file, mode="a+") as f:
        f.writelines(text + '\n')


def plot_chart(total_var, title):
    """打印曲线图"""
    lens = len(total_var)
    plt.figure(figsize=(10, 6), dpi=80)
    plt.plot([x for x in range(lens)], total_var, color="blue", label=title)
    plt.xlabel("Batch")
    plt.ylabel(f"{title}")
    plt.title(title + " With Batch")
    plt.legend()
    name = f"{time.strftime('%Y-%m-%d-%H-%M')}--{title}.jpg"
    plt.savefig(f"D:\\python_code\\paper\\chart\\" + name)
    plt.show()


def adjust_LR(max_acc, now_acc, optimizer, epoch):
    """调整学习率"""
    if now_acc > max_acc:
        optimizer.param_groups[0]['lr'] = LR * (0.95 ** epoch)
    else:
        optimizer.param_groups[0]['lr'] = LR / (0.95 ** epoch)


def cross_valid(train_path, test_path, batch_size, n_splits, tokenizer):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=6)
    train_data, valid_data = [], []
    for train_idx, valid_idx in kf.split(train):
        train_temp = train.iloc[train_idx]
        valid_temp = train.iloc[valid_idx]
        train_temp.index = [x for x in range(len(train_temp))]
        valid_temp.index = [x for x in range(len(valid_temp))]
        train_data.append(train_temp)
        valid_data.append(valid_temp)
    train_iter_list = []
    for data in train_data:
        train_temp = InputDataSet(data, tokenizer, 128)
        train_iter = DataLoader(train_temp, batch_size=batch_size, num_workers=0)
        train_iter_list.append(train_iter)
    valid_iter_list = []
    for data in valid_data:
        valid_temp = InputDataSet(data, tokenizer, 128)
        valid_iter = DataLoader(valid_temp, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)
        valid_iter_list.append(valid_iter)
    test_data = TestInput(test, tokenizer, 128)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)
    return train_iter_list, valid_iter_list, test_iter


def run_train(batch_size, epochs):
    set_seed(GLOBAL_SEED)

    tokenizer = AlbertTokenizer.from_pretrained(now_model)

    train_iter_list, valid_iter_list, test_iter = cross_valid(now_train, JP_test, batch_size, 5, tokenizer)

    info_name = f"{time.strftime('%Y-%m-%d-%H-%M')}.txt"
    final_file = os.path.join("D:\\python_code\\paper\\log", info_name)

    total_time_start = time.time()

    k_result = []
    true_label = []
    for k, (train_iter, valid_iter) in enumerate(zip(train_iter_list, valid_iter_list)):
        model = ALBertForSeq.from_pretrained(
            now_model) if use_model == 'albert' else ALBertAndTextCnnForSeq.from_pretrained(now_model)

        optimizer = AdamW(model.parameters(), lr=LR)

        total_steps = len(train_iter) * epochs

        cache_info(final_file, f"   Train batch size = {batch_size}, K: {k}")
        cache_info(final_file, f"   Total steps = {total_steps}")
        cache_info(final_file, f"   Training Start!")

        file_lens = len(os.listdir(JP_save))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_ratio * total_steps,
            num_training_steps=total_steps)

        # 最大的一次epoch准确率
        max_val_acc = 0.0
        min_val_loss = 999
        # 准确率没上升次数
        repeat_acc = 0

        # 每个batch的loss值
        total_loss = []

        for epoch in track(range(epochs), description="训练中(请勿操作电脑)"):
            total_train_loss = 0
            t0 = time.time()
            # 多GPU跑
            # model = nn.DataParallel(model, device_ids=devices).to(devices[0]) if torch.cuda.device_count() > 1 else model.to(devices[0])
            model = model.to(devices[1])
            model.train()

            for step, batch in enumerate(train_iter):
                input_ids = batch["input_ids"].to(devices[1])
                attention_mask = batch["attention_mask"].to(devices[1])
                token_type_ids = batch["token_type_ids"].to(devices[1])
                labels = batch["labels"].to(devices[1])

                model.zero_grad()

                outputs = model(input_ids, attention_mask, token_type_ids, labels)

                loss = outputs.loss

                total_train_loss += loss.item()

                if step % 100 == 0:
                    total_loss.append(loss.item())

                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_iter)
            now_lr = optimizer.param_groups[0]["lr"]

            writer.add_scalar("loss/train_loss", avg_train_loss, epoch)

            train_time = format_time(time.time() - t0)
            val_time_start = time.time()

            cache_info(final_file,
                       f"====K: {k} || Epoch:[{epoch + 1}/{epochs}] || avg_train_loss={avg_train_loss:.3f} || LR={now_lr}====")
            cache_info(final_file, f"====Training epoch took: {train_time}====")
            cache_info(final_file, "Running Validation...")

            model.eval()
            # 获取验证集的误差和准确度
            avg_val_loss, avg_val_acc = evaluate(model, valid_iter)

            writer.add_scalar("loss/val_loss", avg_val_loss, epoch)
            writer.add_scalar("acc/val_acc", avg_val_acc, epoch)

            # 动态调整学习率
            adjust_LR(max_acc=max_val_acc, now_acc=avg_val_acc, optimizer=optimizer, epoch=epoch)

            writer.add_scalar("learning rate", now_lr, epoch)

            val_time = format_time(time.time() - val_time_start)

            cache_info(final_file,
                       f"====K: {k} || Epoch:[{epoch + 1}/{epochs}] || avg_val_loss={avg_val_loss:.3f} || avg_val_acc={avg_val_acc:.3f}"
                       f" || repeat_time={repeat_acc}====")
            cache_info(final_file, f"====Validation epoch took: {val_time}====")
            cache_info(final_file, "")

            if avg_val_acc <= max_val_acc:
                repeat_acc += 1

            # 若准确率比最大的epoch更好时将模型保存起来
            elif avg_val_acc > max_val_acc or avg_val_loss < min_val_loss:
                # model_to_save = model.module if hasattr(model, 'module') else model
                max_val_acc = avg_val_acc
                min_val_loss = avg_val_loss
                repeat_acc = 0

                model_to_save = model
                output_dir = now_save
                output_name = f"{file_lens}-model.bin"
                output_model_file = os.path.join(output_dir, output_name)
                torch.save(model_to_save.state_dict(), output_model_file)
                with open("./config.json", mode='r') as f1:
                    ps = f1.read()
                    ps = json.loads(ps)
                    ps["my_model"]["MY_MODEL_PATH"] = output_model_file
                    ps["my_model"]["MY_MODEL_TYPE"] = "japanese" if now_model == JP_model else "chinese"
                    ps["my_model"]["MY_MODEL_NAME"] = output_name
                    ps["use_model"] = "albert" if use_model == "albert" else "albert_textcnn"
                with open("./config.json", mode='w') as f2:
                    json.dump(ps, f2, indent=2)

                print("Model saved!")

            # 若准确率连续三次都没有提升则停止训练
            if repeat_acc == 3:
                break

        lst_prob, lst_true = my_prediction(model, test_iter, "D:\\python_code\\paper\\data\\test_label2.csv", info_name, devices[1])
        k_result.append(lst_prob)
        true_label = lst_true

        plot_chart(total_loss, "loss")
        print(f"max acc is : {max_val_acc}")

        out = [
            [params["my_model"]["MY_MODEL_NAME"],
             params["use_model"],
             "chinese" if now_model == ZH_model else "japanese",
             str(now_train).split('\\')[-1],
             str(now_val).split('\\')[-1],
             LR,
             GLOBAL_SEED,
             max_val_acc]
        ]
        df = pd.DataFrame(out)
        df.to_csv("../exp_res.csv", mode='a', header=False, index=False)

    avg_prediction(k_result, true_label)
    cache_info(final_file, "")
    cache_info(final_file, "   Training Completed!")
    print(f"Total train time: {format_time(time.time() - total_time_start)}")


def evaluate(model, val_iter):
    """计算验证集的误差和准确率"""
    total_val_loss = 0
    corrects = []
    for batch in val_iter:
        # 从迭代器中取出每个批次
        input_ids = batch["input_ids"].to(devices[1])
        attention_mask = batch["attention_mask"].to(devices[1])
        token_type_ids = batch["token_type_ids"].to(devices[1])
        labels = batch["labels"].to(devices[1])

        # 验证集的outputs不参与训练集后续的梯度计算
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids, labels)

        # 获取该批次中所有样本的所有分类中的最大值，将最大值变为1，其余变为0
        logits = torch.argmax(outputs.logits, dim=1)
        # 将预测值不参与后续训练集的梯度计算
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to("cpu").numpy()
        # 求出该批次的准确率
        corrects.append((preds == labels_ids).mean())

        loss = outputs.loss
        # 累加损失
        # total_val_loss += loss.mean().item()
        total_val_loss += loss.item()

    # 求出平均损失
    avg_val_loss = total_val_loss / len(val_iter)
    # 求出平均准确率
    avg_val_acc = np.mean(corrects)

    return avg_val_loss, avg_val_acc


if __name__ == '__main__':
    run_train(16, 20)
    writer.close()
