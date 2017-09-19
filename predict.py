# coding=utf-8

import os
import pandas as pd
import jieba
import pickle
import numpy as np
import re
from functools import reduce

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import util.preprocess as pr
from sklearn import metrics

pattern = re.compile(
    u"[/~*'зД=つ\-Дノ→\+·②⑥:°【［•”˙×ε○↖\\、↗∧⊙⊥⑨┌╯▽罒】☆⌒∩〃￥〕ㄒㄥ〔`\[\"\?\]ò\*ω￥∀Σπ●《》……^≦～－——？()\.／：；‘“、）（……％?¥＃/#@$%^&]")
splitPattern = re.compile(u"[，,!！~。;]")
# jieba.load_userdict("dict.txt")


def cleanData(toclean):
    rt = toclean.replace(u"?", "")
    rt = re.sub(pattern, "", rt)
    rt = re.sub(splitPattern, " ", rt)
    return list(filter(lambda x: len(x) > 0, rt.split(" ")))

def ngramHelper2(rawstr, num):
    # splits = [i for i in jieba.cut(str(rawstr), cut_all=False)]
    splits = rawstr
    result=''
    if len(splits) == 1:
        result =splits[0]
    else:
        if num == 1:
            result=" ".join(splits)
        elif num == 2:
            resulttmp = [splits[i] + splits[i + 1] for i in range(0, len(splits) - 1)]
            result = " ".join(resulttmp)
        elif num == 3:
            resulttmp = [splits[i] + splits[i + 1] + splits[i + 2] for i in range(0, len(splits) - 1)]
            result = " ".join(resulttmp)
    return result


def ngramExtract(raw, num):
    if len(raw) == 0:
        return "##"
    else:
        # return list(reduce(lambda x1, x2: x1 + x2, map(lambda x: ngramHelper(x, num), raw)))
        # print(list(map(lambda x: ngramHelper2(x, num), raw)))
        return " ".join(list(map(lambda x: ngramHelper2(x, num), raw)))


def featureExtract(vectorizer, testdata):
    testbunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    for line in testdata:
        s = line.split("\t")
        if len(s) == 2:
            dataline = cleanData(s[1])
            linelist = ngramExtract(dataline, 2)
            testbunch.target_name.extend(s[0])
            testbunch.label.append(s[0])
            testbunch.filenames.append(s[0])
            testbunch.contents.append(str(linelist))
    testTfidfspace = Bunch(target_name=testbunch.target_name, label=testbunch.label, filenames=testbunch.filenames,
                           tdm=[],
                           vocabulary={})
    testTfidfspace.tdm = vectorizer.fit_transform(testbunch.contents)
    return testTfidfspace.tdm

def loadModel(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
        return model

def manager(testPath,resultPath):
    testData = pr.readData(testPath)
    model=loadModel('model.pickle')
    vectorizer=loadModel('vectorizer.pickle')
    test_d=featureExtract(vectorizer, testData)
    pred=model.predict(test_d)
    result=combineList(pred,testData)
    pr.saveData(resultPath,result)

def combineList(list1,list2):
    a = np.mat(list1)
    b = np.row_stack((a, list2)).T
    c=b.tolist()
    d = list(map(lambda x: reduce(lambda x1, x2: str(x1) + '\t' + str(x2), x),c))
    return d

if __name__ == '__main__':
    manager( './data/testdata.txt','./data/result.txt')
