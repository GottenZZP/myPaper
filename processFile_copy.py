# coding:utf-8

from email import header
import numpy as np
import pandas as pd
import time
import os
import re
from regex import P
import torch
import warnings
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import logging
from sklearn.model_selection import train_test_split
from sklearn import datasets
from transformers.utils.notebook import format_time

logging.set_verbosity_warning()
# 取消报错
warnings.filterwarnings('ignore')

class ProcessFile():
    """文件处理类"""
    def __init__(self, file_path) -> None:
        self.root_file_path = file_path

    def process_file(self, file_path):
        """获取所有的标题及其出现次数"""
        if not os.path.exists(file_path):
            print("The file does not exist!")
            return None
        
        time_start = time.time() 

        total_title = dict()

        pattern = r'.*?title ="(?P<name>.*?)"> '
        obj = re.compile(pattern, re.S)

        with open(file_path, mode='r', encoding='utf-8') as f:
            num = 0

            for line in f.readlines():
                title = obj.search(line)
                if title is not None:
                    if title.group("name") not in total_title.keys():
                        total_title[title.group("name")] = 1
                    else:
                        total_title[title.group("name")] += 1
                
                if len(total_title) % 100 == 0 and num != len(total_title):
                    print(f"Saved : [{len(total_title)}]")
                    num = len(total_title)
                
        
        print(f"Total time : {format_time(time.time() - time_start)} !")
        return total_title

    def get_title_info(self, file_path, valid_len):
        """筛选掉不合格的标题"""
        if not os.path.exists(file_path):
            print("The file does not exist!")
            return None
        
        pattern1 = r'[年月日村郡県駅学]'
        obj1 = re.compile(pattern1, re.S)

        pattern2 = r'\s*[a-zA-Z]+\s*[a-zA-Z]*\s*'
        obj2 = re.compile(pattern2, re.S)

        pattern3 = r'[\d]+\s*[a-zA-Z]*\s*'
        obj3 = re.compile(pattern3, re.S)

        df = pd.read_csv(file_path, sep=',')
        final_file = dict()
        title, num = df["title"], df["num"]
        for t, n in zip(title, num):
            if int(n) >= valid_len and obj1.search(t) is None and obj2.search(t) is None and obj3.search(t) is None:
                final_file[t] = n
        
        return final_file
    
    def make_data_set(self, final_file):
        """获取每个标题的内容，以键值对的方式返回"""
        if not os.path.exists(final_file):
            print("The file does not exist!")
            return None

        df = pd.read_csv(final_file)
        titles = df["title"].to_list()
        title_info = dict()
        total_len = 0

        # (.*?)</ doc > 
        pattern1 = r'.*?title ="(?P<title>.*?)">'
        obj1 = re.compile(pattern1, re.S)

        pattern2 = r'</ doc >'
        obj2 = re.compile(pattern2, re.S)

        with open("D:\\python_code\\paper\\corpus\\wiki_wakati.txt", mode='r', encoding="utf-8") as f:
            flag = False
            s = ""
            num = 0
            for line in f.readlines():
                temp1 = obj1.search(line)
                temp2 = obj2.search(line)
                
                if flag:
                    s += str(line).strip()

                if temp1 is not None:
                    title = temp1.group("title")
                    if title in titles:
                        flag = True
                
                if temp2 is not None:
                    if title not in title_info and flag:
                        title_info[title] = s
                    elif title in title_info and flag:
                        if s > title_info[title]:
                            title_info[title] = s
                    s = ""
                    flag = False

                if len(title_info) % 10 == 0 and num != len(title_info):
                    print(f"Saved : [{len(title_info)}]")
                    num = len(title_info)
        
        for k, v in title_info.items():
            total_len += len(v)

        self.save_file(title_info, "corpus", "dataSet1")
        return total_len // len(title_info)

    def final_process(self, file_path, max_len, name):  
        if not os.path.exists(file_path):
            print("The file does not exist!")
            return None
        total_len = 0
        df = pd.read_csv(file_path)
        title, context = df["title"], df["num"]

        final_version = dict()

        for t, c in zip(title, context):
            if len(c) >= max_len:
                total_len += len(c)
                final_version[t] = c[len(t): -10]

        self.save_file(final_version, "corpus", name)
        return total_len // len(final_version), len(final_version)


    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")


    def split_sentence(self, file_path, sent_min_len=5):
        """切分句子"""
        if not os.path.exists(file_path):
            print("The file does not exist!")
            return None

        df = pd.read_csv(file_path)
        titles, context = df["title"].to_list(), df["num"].to_list()

        dataSet = []
        total_lens = 0
        split_lens = dict()

        for t, n in zip(titles, context):
            num = 0
            sents = self.cut_sent(str(n))
            if len(sents) > 1:
                for sent in sents:
                    if len(sent) > sent_min_len:
                        dataSet.append([t, sent])
                        total_lens += len(sent)
            elif len(sents) == 1:
                if len(sents[0] > sent_min_len):
                    dataSet.append([t, sents[0]])
                    total_lens += len(sents[0])

            num += len(sents)
            split_lens[t] = num
            print(t, ': ', num)

        
        df = pd.DataFrame(dataSet, columns=["label", "text"])
        self.save_file(df, "data", "split_data1")
        self.save_file(split_lens, "data", "split_lens1")

        return total_lens // len(dataSet)    


    def save_file(self, total_title, rela_path, file_name):
        """保存数据"""
        if isinstance(total_title, dict):
            df = pd.DataFrame(pd.Series(total_title), columns=['num'])
            df = df.reset_index().rename(columns={'index': 'title'})
        else:
            df = total_title

        full_path = os.path.join(self.root_file_path, rela_path)

        df.to_csv(full_path + f"\\{file_name}.csv", sep=',', index=False)
        print("Saved successfully!")


    def norm_data(self, file_path):
        """将数据规范化"""
        if not os.path.exists(file_path):
            print("The file does not exist!")
            return None

        df = pd.read_csv(file_path)
        labels = df["label"]
        texts = df["text"]

        output = []

        for l, t in zip(labels, texts):
            l = l.strip().replace(' ', '')
            t = t.strip()
            output.append([l, t])
        
        df = pd.DataFrame(output, columns=["label", "num"])
        self.save_file(df, "data", "split_data_norm1")


    def split_train_val(self, file_path, rite):
        if not os.path.exists(file_path):
            print("The file does not exist!")
            return None

        df = pd.read_csv(file_path)
        labels, texts = df["label"], df["text"]

        train_data, val_data, train_label, val_label = train_test_split(texts, labels, shuffle=True, test_size=rite, random_state=6)

        train = zip(train_data, train_label)
        val = zip(val_data, val_label)

        train_df = pd.DataFrame(train, columns=["text", "label"])
        val_df = pd.DataFrame(val, columns=["text", "label"])

        self.save_file(train_df, "data", "train")
        self.save_file(val_df, "data", "val")

    def split_train_test(self, file_path, rite):
        if not os.path.exists(file_path):
            print("The file does not exist!")
            return None

        df = pd.read_csv(file_path)
        labels, texts = df["label"], df["text"]

        train_data, test_data, train_label, test_label = train_test_split(texts, labels, shuffle=True, test_size=rite, random_state=6)

        train = zip(train_data, train_label)

        train_df = pd.DataFrame(train, columns=["text", "label"])
        test_df = pd.DataFrame(test_data, columns=["text"])
        test_label_df = pd.DataFrame(test_label, columns=["label"])

        self.save_file(train_df, "data", "trainB")
        self.save_file(test_df, "data", "test")
        self.save_file(test_label_df, "data", "test_label")

    def to_idx(self, file_path):
        df = pd.read_csv(file_path)
        labels = df["label"]
        idx = dict()
        for i, l in enumerate(labels):
            idx[i] = l
        self.save_file(idx, "data", "idx")

    def label_to_idx(self, file_path1, file_path2):
        if not os.path.exists(file_path1):
            print("The file does not exist!")
            return None

        df = pd.read_csv(file_path1)
        df1 = pd.read_csv(file_path2)
        labels, idx = df["label"], df["idx"]
        labels1, text1 = df1["label"], df1["text"]
        
        label_to_idx = dict()
        for l, i in zip(labels, idx):
            label_to_idx[l] = i
        
        out = []
        for l, t in zip(labels1, text1):
            out.append([t, label_to_idx[l]])
        
        df2 = pd.DataFrame(out, columns=["text", "label"])
        self.save_file(df2, "data", "val2")
        


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


class InputDataSet():

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
            'text':text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'labels':labels
        }

class TestInput():

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
            'text':text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
        }


if __name__ == "__main__":
    # file_path = "D:\\python_code\\paper\\corpus\\wiki_wakati.txt"
    # total_title = process_file(file_path)
    # total_title = get_title_info("D:\\python_code\\paper\\corpus\\allTitle.csv", 3)
    # save_file(total_title, "validTitle3")
    # print(len(total_title))   
    # mean_len = make_data_set("D:\\python_code\\paper\\corpus\\validTitle3.csv")
    # print(mean_len)
    # mean_len, title_len = final_process("D:\\python_code\\paper\\corpus\\dataSet.csv", 120000, "final_data_120000")
    # print(mean_len, title_len)

    # mean_len = split_sentence("D:\\python_code\\paper\\corpus\\final_data_120000.csv")
    # print(mean_len)
    
    # norm_data("D:\\python_code\\paper\\data\\split_data.csv")

    obj = ProcessFile("D:\\python_code\\paper")
    # obj.norm_data("D:\\python_code\\paper\\data\\split_lens.csv")
    # obj.label_to_idx("D:\\python_code\\paper\\data\\idx.csv", "D:\\python_code\\paper\\data\\test_label.csv")
    # obj.make_data_set("D:\\python_code\\paper\\corpus\\validTitle3.csv")
    # mean_len, title_len = obj.final_process("D:\python_code\paper\corpus\dataSet1.csv", 60000, "final_data_60000")
    # print(mean_len, title_len)
    # mean_len = obj.split_sentence("D:\\python_code\\paper\\corpus\\final_data_60000.csv", 10)
    # print(mean_len)
    # obj.norm_data("D:\\python_code\\paper\\data\\split_lens1.csv")
    # obj.norm_data("D:\python_code\paper\data\split_data1.csv")
    # obj.split_train_val("D:\python_code\paper\data\split_data_norm1.csv", 0.2)
    # obj.split_train_test("D:\\python_code\\paper\\data\\train.csv", 0.1)
    # obj.to_idx("D:\python_code\paper\data\split_lens_norm1.csv")
    # obj.label_to_idx("D:\\python_code\\paper\\data\\idx.csv", "D:\\python_code\\paper\\data\\test_label.csv")
    obj.label_to_idx("D:\\python_code\\paper\\data\\idx.csv", "D:\\python_code\\paper\\data\\val.csv")