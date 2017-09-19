#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import jieba
import pickle
import numpy as np
import re
from functools import reduce
from  sklearn.metrics import classification_report, confusion_matrix
import sys



# arguments = sys.argv
# try:
#     if len(arguments) == 2:
#         print (u"开始读原文件.......")
#         data = pd.read_excel(arguments[1])
#     else:
#         raise Exception(u"参数设置错误")
# except Exception as e:
#     print (u"大哥运行程序的方式错了,需要把文件名当成参数传进去")
#     print (u"如\n> python run.py 原文件.xlsx")
#     sys.exit()

data = pd.read_excel("./data/test.xlsx")
try:
    textdata = data[u"内容"]
except:
    print (u"大哥原文件[内容]这一列找不到")
    sys.exit()

pattern = re.compile(u"[/~*'зД=つ\-Дノ→\+·②⑥:°【［•”˙×ε○↖\\、↗∧⊙⊥⑨┌╯▽罒】☆⌒∩〃￥〕ㄒㄥ〔`\[\"\?\]ò\*ω￥∀Σπ●《》……^≦～－——？()\.／：；‘“、）（……％?¥＃/#@$%^&]")
splitPattern = re.compile(u"[，,!！~。;]")
jieba.load_userdict("dict.txt")

def cleanData(toclean):
    rt = toclean.replace(u"?", "")
    rt = re.sub(pattern, "", rt)
    rt = re.sub(splitPattern, " ", rt)
    if len(rt) > 0:
        return rt.split(" ")
    # return filter(lambda x: len(x) > 0, rt.split(" "))

def bigramHelper(rawstr):
    splits = [ i for i in jieba.cut(str(rawstr), cut_all=False)]
    if len(splits) == 1:
        return [splits[0]]
    else:
        print([ splits[i] + splits[i+1] for i in range(0, len(splits)-1)])
        return [ splits[i] + splits[i+1] for i in range(0, len(splits)-1)]
    
def bigramExtract(raw):
    '''提取Bigram特征'''
    if len(raw) == 0:
        return [u"##"]
    else:
        return list(set(reduce(lambda x1, x2: x1+x2, map(lambda x: bigramHelper(x), raw))))

corpus = []
count = 0
print (u"数据清洗.......")

for line in textdata:
    if type(line) != np.unicode:
        line = str(line)
    line=cleanData(line)
    linelist=bigramExtract(line)
    corpus.append(" ".join(linelist))
    count += 1
    if count % 10000 == 0:
        print (count)

textdata = np.array(data[u"内容"])

fil = set()
with open("ngramStat.txt", "rb") as f:
    for line in f:
        tmp = line.strip().split("\t")
        if len(tmp) == 1:
            continue
        if int(tmp[0]) > 5:
            fil.add(tmp[1])

with open("vectorizer.pickle", "rb") as f:
    vectorizer = pickle.load(f)

newCorpus = []
for line in corpus:
    tmp = line.split(" ")
    tL = [i for i in tmp if i in fil]
    if len(tL) == 0:
        newCorpus.append("**")
    else:
        newCorpus.append(" ".join(tL))

X = vectorizer.fit_transform(newCorpus)

with open("model.pickle", "rb") as f:
    classif = pickle.load(f)

print (u"开始预测。。。。。。。。。\n")

prob = classif.predict_proba(X)

def cal(threshold):
    return map(lambda x: -1 if x else 1, (prob[:,0] - prob[:,1]) > threshold)


predict = cal(0.3)
data["predict"] = predict

print (u"预测完成，开始写文件.........嗯，有点慢\n")

outputFileName = "output.xlsx"
data.to_excel(outputFileName, index=False)

print (u"输出文件名称： " + outputFileName)