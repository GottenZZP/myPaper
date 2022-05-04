import http.client
import hashlib
from json import tool
import urllib
import random
import json
import time
import pandas as pd
from pip._vendor.distlib.compat import raw_input
from pyparsing import col
from sklearn.utils import shuffle

def translate(q, lang='zh'):
    appid = '20220321001133615'
    secretKey = 'pgbdPDuQpual8Ssw8VMb'

    httpClient = None
    myurl = '/api/trans/vip/translate'
    
    fromLang = 'auto'
    toLang = lang
    salt = random.randint(32768, 65536)
    
    # q = raw_input("input word or sentence: ")
    sign = appid + str(q) + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    try:
        myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + \
            toLang + '&salt=' + str(salt) + '&sign=' + sign
    except TypeError:
        return None
    
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        return str(result['trans_result'][0]['dst'])
    except Exception as e:
        return e
    finally:
        if httpClient:
            httpClient.close()

def get_mean(file_path):
    df = pd.read_csv(file_path)
    labels = df["label"]
    num = dict()
    for i in labels:
        if i not in num:
            num[i] = 1
        else:
            num[i] += 1
    return num, len(labels) // len(num)

def dataEnhance(file_path):
    df = pd.read_csv(file_path)
    labels = df["label"]
    texts = df['text']
    nums, meams = get_mean(file_path)
    l_t = list(zip(texts, labels))
    l_t.sort(key=lambda x: x[1])
    saved = []
    zimu = {0: 'zh', 1: 'en', 2: 'yue', 3: 'kor', 4: 'fra', 5: 'spa', 6: 'th', 7: 'ara', 8: 'ru', 9: 'pt', 10: 'de', 11: 'it'}
    for l in range(23, 31):
        n = 0
        a = sum([nums[x] for x in range(l)])
        b = a + nums[l]
        dataSet = []
        while nums[l] < meams:
            r = random.randint(a + 3, b - 3)
            r1 = random.randint(0, 11)
            if l_t[r][1] != l:
                continue
            temp = translate(l_t[r][0], zimu[r1])
            if not temp or temp == r"'trans_result'" or temp is None:
                continue
            time.sleep(1)
            text = translate(temp, 'jp')
            if not text or text == r"'trans_result'" or text is None:
                continue
            time.sleep(1)
            dataSet.append((text, l))
            nums[l] += 1
            n += 1
            print(f"label: {l}, num: [{n}]\nOrign: {l_t[r][0]}\nEnhance: {text}\n")
        saved.append(l)
        # l_t.extend(dataSet)
        # random.shuffle(l_t)
        res = pd.DataFrame(dataSet, columns=["text", "label"])
        res.to_csv("D:\python_code\paper\data\\val_enhance.csv", index=False, mode='a')
        print("Saved ", saved)

def dataSplice(file_path, file_path1):
    df = pd.read_csv(file_path)
    df1 = pd.read_csv(file_path1)
    labels = df["label"]
    texts = df['text']
    labels1 = df1["label"]
    texts1 = df1['text']
    l_t = list(zip(texts, labels))
    l_t1 = list(zip(texts1, labels1))
    l_t.extend(l_t1)
    random.shuffle(l_t)
    out = pd.DataFrame(l_t, columns=["text", "label"])
    out.to_csv("D:\python_code\paper\data\\val4.csv", index=False)

def enhanceHandle(file_path):
    df = pd.read_csv(file_path)
    labels = df["label"]
    texts = df['text']
    l_t = list(zip(texts, labels))
    data = []
    n = 0
    for i in range(len(l_t)):
        if l_t[i][0] == "'trans_result'" or l_t[i][0] == "text":
            n += 1
            continue
        data.append(l_t[i])
    out = pd.DataFrame(data, columns=["text", "label"])
    out.to_csv("D:\python_code\paper\data\\val_enhance.csv", index=False)
    print(n)

def underSampling(file_path):
    df = pd.read_csv(file_path)
    labels = df["label"]
    texts = df['text']
    nums, _ = get_mean(file_path)
    l_t = list(zip(texts, labels))
    for l in range(30):
        while nums[l] > 600:
            # l_t.
            pass


if __name__ == "__main__":
    # dataEnhance("D:\python_code\paper\data\\val3.csv")
    means, _ = get_mean("D:\python_code\paper\data\\train5.csv")
    print(means)
    # dataSplice("D:\python_code\paper\data\\val3.csv", "D:\python_code\paper\data\\val_enhance.csv")
    # enhanceHandle("D:\python_code\paper\data\\val_enhance.csv")