from operator import mod
from unicodedata import decimal
import numpy as np
from torch import nn
import time
import os
import torch
import logging
import random
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup,  WEIGHTS_NAME, CONFIG_NAME
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time
from modeling_copy import BertForSeq, TextCNN
from processFile import InputDataSet, read_file
from d2l import torch as d2l
from rich.progress import track
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

devices = d2l.try_all_gpus()
# 中文参数
ZH_model = "D:\\python_code\\paper\\models\\bert-base-chinese"
ZH_train = "D:\\python_code\\paper\\corpus\\chinese\\train2.csv"
ZH_val = "D:\\python_code\\paper\\corpus\\chinese\\val2.csv"
ZH_save = "D:\\python_code\\paper\\models\\chinese"
# 日本参数
JP_model = "D:\python_code\paper\models\\bert-base-japanese"
JP_train = "D:\python_code\paper\data\\train5.csv"
JP_val = "D:\python_code\paper\data\\val4.csv"
JP_save = "D:\python_code\paper\models\japanese\clean_model_textcnn"
# 当前使用模型
now_model = JP_model
# 全局随机种子
GLOBAL_SEED = 6
# 线程数
GLOBAL_WORKER_ID = 4
num_workers = 4
# tensorboard路径
log_path = "D:\python_code\paper\summary\\"
# 学习率
LR = 5e-5
# 学习率预热
warm_up_ratio = 0.1
# 句子最大长度
SEQ_MAX_LEN = 512

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


def train(batch_size, epochs):
    set_seed(GLOBAL_SEED)

    # 加载预训练
    model = BertForSeq.from_pretrained(now_model)

    # 加载训练集与验证集
    train = read_file(JP_train)
    val = read_file(JP_val)

    tokenizer = BertTokenizer.from_pretrained(now_model)

    train_data = InputDataSet(train, tokenizer, SEQ_MAX_LEN)
    val_data = InputDataSet(val, tokenizer, SEQ_MAX_LEN)

    train_iter = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)
    val_iter = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)

    optimizer = AdamW(model.parameters(), lr=LR)
    # lr_change = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', min_lr=1e-5)
    
    total_steps = len(train_iter) * epochs
    
    # 学习率预热
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warm_up_ratio * total_steps, 
        num_training_steps=total_steps)
    
    total_time_start = time.time()

    info_name = f"{time.strftime('%Y-%m-%d-%H-%M')}.txt"
    final_file = os.path.join("D:\\python_code\\paper\\log", info_name)

    cache_info(final_file, f"   Train batch size = {batch_size}")
    cache_info(final_file, f"   Total steps = {total_steps}")
    cache_info(final_file, f"   Training Start!")

    # 上一次的验证集准确率
    max_val_acc = 0.0
    min_val_loss = 999
    repeat_acc = 0

    # 文件夹内文件个数
    file_lens = len(os.listdir(JP_save))

    total_loss = []

    for epoch in track(range(epochs), description="Training..."):
        total_train_loss = 0
        t0 = time.time()
        # 多GPU跑
        # model = nn.DataParallel(model, device_ids=devices).to(devices[0]) if torch.cuda.device_count() > 1 else model.to(devices[0])
        model = model.to(devices[0])
        model.train()
        for step, batch in enumerate(train_iter):
            # 解包
            input_ids = batch["input_ids"].to(devices[0])
            attention_mask = batch["attention_mask"].to(devices[0])
            token_type_ids = batch["token_type_ids"].to(devices[0])
            labels = batch["labels"].to(devices[0])

            # writer.add_graph(model, (input_ids, attention_mask, token_type_ids, labels))

            model.zero_grad()
            
            # 获取输出
            outputs = model(input_ids, attention_mask, token_type_ids, labels)

            loss = outputs.loss
            # 将每轮的损失累加起来
            total_train_loss += loss.item()
            if step % 100 == 0:
                total_loss.append(loss.item())                

            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_iter)
        writer.add_scalar("loss/train_loss", avg_train_loss, epoch)

        now_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("learning rate", now_lr, epoch)

        train_time = format_time(time.time() - t0)

        val_time_start = time.time()

        cache_info(final_file, f"====Epoch:[{epoch + 1}/{epochs}] || avg_train_loss={avg_train_loss:.3f} || LR={now_lr:.6f}====")
        cache_info(final_file, f"====Training epoch took: {train_time}====")
        cache_info(final_file, "Running Validation...")

        model.eval()
        # 获取验证集的误差和准确度
        avg_val_loss, avg_val_acc = evaluate(model, val_iter)

        # 当准确率没有提升时候动态调整学习率
        # lr_change.step(avg_val_acc)

        writer.add_scalar("loss/val_loss", avg_val_loss, epoch)
        writer.add_scalar("acc/val_acc", avg_val_acc, epoch)

        val_time = format_time(time.time() - val_time_start)

        cache_info(final_file, f"====Epoch:[{epoch + 1}/{epochs}] || avg_val_loss={avg_val_loss:.3f} || avg_val_acc={avg_val_acc:.3f}====")
        cache_info(final_file, f"====Validation epoch took: {val_time}====")
        cache_info(final_file, "")

        if avg_val_acc <= max_val_acc:
            repeat_acc += 1

        # 若准确率比上次epoch更好时将模型保存起来
        elif avg_val_acc > max_val_acc:
            # model_to_save = model.module if hasattr(model, 'module') else model
            max_val_acc = avg_val_acc
            min_val_loss = avg_val_loss
            repeat_acc = 0

            model_to_save = model
            output_dir = JP_save
            output_name = f"{file_lens}-model.bin"
            output_model_file = os.path.join(output_dir, output_name)
            # output_config_file = os.path.join(output_dir, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            # model_to_save.config.to_json_file(output_config_file)
            # tokenizer.save_vocabulary(output_dir)
            print("Model saved!")
        
        # 若准确率连续三次都没有提升则停止训练
        # if repeat_acc == 3:
        #     break

    cache_info(final_file, "")
    cache_info(final_file, "   Training Completed!")
    print(f"Total train time: {format_time(time.time() - total_time_start)}")
    plot_chart(total_loss, "loss")


def evaluate(model, val_iter):
    """计算验证集的误差和准确率"""
    total_val_loss = 0
    corrects = []
    for batch in val_iter:
        # 从迭代器中取出每个批次
        input_ids = batch["input_ids"].to(devices[0])
        attention_mask = batch["attention_mask"].to(devices[0])
        token_type_ids = batch["token_type_ids"].to(devices[0])
        labels = batch["labels"].to(devices[0])

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
    name =  f"{time.strftime('%Y-%m-%d-%H-%M')}.jpg"
    plt.savefig(f"D:\\python_code\\paper\\chart\\" + name)
    plt.show()

if __name__ == "__main__":
    train(16, 20)
    writer.close()