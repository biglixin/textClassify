# coding=utf-8
from functools import reduce

import re
import numpy as np
from gensim.models import word2vec
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch

import util.preprocess as pr
# sentences=word2vec.LineSentence('D:/Users/lianglx5/PycharmProjects/textClassfiy/data/w2vdata.txt')
pattern = re.compile(
    u"[\s\ta-zA-Z0-9，,!！~。;/~*'зД=つ\-Дノ→\+·②⑥:°【［•”˙×ε○↖\\、↗∧⊙⊥⑨┌╯▽罒】☆⌒∩〃￥〕ㄒㄥ〔`\[\"\?\]ò\*ω￥∀Σπ●《》……^≦～－——？()\.／：；‘“、）（……％?¥＃/#@$%^&.!?]")

def cleanData(toclean):
    rt = toclean.replace(u"?", "")
    rt = re.sub(pattern, "", rt)
    rt=rt.replace(".","")
    return rt


if __name__ == '__main__':
    # pr.splitSentienceToChar("D:/Users/lianglx5/PycharmProjects/textClassfiy/data/char2vecData.txt","D:/Users/lianglx5/PycharmProjects/textClassfiy/data/char2vecInput.txt")
    sentences = pr.readData('D:\\Users\\lianglx5\\PycharmProjects\\textClassify\\data\\char2vecData.txt')
    l = []
    for i in sentences:
        l.append(" ".join(cleanData(i)))
    model = word2vec.Word2Vec(sentences, size=100, min_count=5)
    model.save('word2vec_model')

