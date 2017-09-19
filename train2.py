# coding=utf-8

import os
import pandas as pd
import jieba
import pickle
import numpy as np
import re
from functools import reduce

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import util.preprocess as pr
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from  sklearn.metrics import classification_report, confusion_matrix
import sys
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1)]
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()]

pattern = re.compile(
    u"[/~*'зД=つ\-Дノ→\+·②⑥:°【［•”˙×ε○↖\\、↗∧⊙⊥⑨┌╯▽罒】☆⌒∩〃￥〕ㄒㄥ〔`\[\"\?\]ò\*ω￥∀Σπ●《》……^≦～－——？()\.／：；‘“、）（……％?¥＃/#@$%^&]")
splitPattern = re.compile(u"[，,!！~。;]")
jieba.load_userdict("dict.txt")


def cleanData(toclean):
    rt = toclean.replace(u"?", "")
    rt = re.sub(pattern, "", rt)
    rt = re.sub(splitPattern, " ", rt)
    return list(filter(lambda x: len(x) > 0, rt.split(" ")))

def ngramHelper(rawstr, num):
    splits = [i for i in jieba.cut(str(rawstr), cut_all=False)]
    result = []
    if len(splits) == 1:
        return [splits[0]]
    else:
        if num == 1:
            result = [splits[i] for i in range(0, len(splits) - 1)]
        elif num == 2:
            result = [splits[i] + splits[i + 1] for i in range(0, len(splits) - 1)]
        elif num == 3:
            result = [splits[i] + splits[i + 1] + splits[i + 2] for i in range(0, len(splits) - 1)]
        return result
def ngramHelper2(rawstr, num):
    # splits = [i for i in jieba.cut(str(rawstr), cut_all=False)]
    splits=rawstr
    result=''
    if len(splits) == 1:
        return splits[0]
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
        return list(reduce(lambda x1, x2: x1 + x2, map(lambda x: ngramHelper(x, num), raw)))
        # print(list(map(lambda x: ngramHelper2(x, num), raw)))
        # print(" ".join(list(map(lambda x: ngramHelper2(x, num), raw))))
        return " ".join(list(map(lambda x: ngramHelper2(x, num), raw)))


def featureExtract(traindata, testdata):
    trainbunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    testbunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    for line in traindata:
        s = line.split("\t")
        if len(s) == 2:
            dataline = cleanData(s[1])
            linelist = ngramExtract(dataline, 2)
            trainbunch.target_name.extend(s[0])
            trainbunch.label.append(s[0])
            trainbunch.filenames.append(s[0])
            trainbunch.contents.append(linelist)

    for line in testdata:
        s = line.split("\t")
        if len(s) == 2:
            dataline = cleanData(s[1])
            linelist = ngramExtract(dataline, 2)
            testbunch.target_name.extend(s[0])
            testbunch.label.append(s[0])
            testbunch.filenames.append(s[0])
            testbunch.contents.append(linelist)
    return trainbunch.contents, trainbunch.label, testbunch.contents, testbunch.label


def manager(trainPath, testPath):
    trainData = pr.readData(trainPath)
    testData = pr.readData(testPath)
    for name, clfname in zip(names, classifiers):
        clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', clfname),
                        ])
        train_d, train_l, test_d, test_l = featureExtract(trainData, testData)
        clf = clf.fit(train_d, train_l)
        predicted = clf.predict(test_d)
        print(name+ ' predict info:')
        print(metrics.classification_report(test_l, predicted))
        # joblib.dump(clf, name + ".pkl")

if __name__ == '__main__':
    # pr.spiltData('./data/22.txt',0.8)
    manager('./data/traindata.txt', './data/testdata.txt')
