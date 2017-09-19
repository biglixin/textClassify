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

CHARTOVEC_SIZE=100

pattern = re.compile(
    u"[，,!！~。;/~*'зД=つ\-Дノ→\+·②⑥:°【［•”˙×ε○↖\\、↗∧⊙⊥⑨┌╯▽罒】☆⌒∩〃￥〕ㄒㄥ〔`\[\"\?\]ò\*ω￥∀Σπ●《》……^≦～－——？()\.／：；‘“、）（……％?¥＃/#@$%^&]")

def cleanData(toclean):
    rt = toclean.replace(u"?", "")
    rt = re.sub(pattern, "", rt)
    return rt
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


def getSentenceCharVec(s):
    # li=[new_model[i] for i in s]
    li=[]
    for i in s:
        try:
            li.append(new_model[i])
        except:
            li.append(np.zeros((1,CHARTOVEC_SIZE)))
    s=list(reduce(lambda  x1, x2: x1 + x2,li))
    s=list(map(lambda x:x/len(s),s))
    return s

def featureExtract(traindata, testdata):
    trainbunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    testbunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    for line in traindata:
        s = line.split("\t")
        if len(s) == 2:
            dataline = cleanData(s[1])
            linelist = getSentenceCharVec(dataline)
            trainbunch.target_name.extend(s[0])
            trainbunch.label.append(s[0])
            trainbunch.filenames.append(s[0])
            trainbunch.contents.append(linelist)
    for line in testdata:
        s = line.split("\t")
        if len(s) == 2:
            dataline = cleanData(s[1])
            linelist = getSentenceCharVec(dataline, 2)
            testbunch.target_name.extend(s[0])
            testbunch.label.append(s[0])
            testbunch.filenames.append(s[0])
            testbunch.contents.append(str(linelist))
    return trainbunch.contents, trainbunch.label, testbunch.contents, testbunch.label

def manager(trainPath, testPath):
    trainData = pr.readData(trainPath)
    testData = pr.readData(testPath)
    train_d, train_l, test_d, test_l = featureExtract(trainData, testData)
    for name, clf in zip(names, classifiers):
        clf.fit(train_d, train_l)
        pred = clf.predict(test_d)
        print(name + ' predict info:')
        print(metrics.classification_report(test_l, pred))
if __name__ == '__main__':
    # print(getSentenceCharVec('臭划重点就是'))
    new_model = word2vec.Word2Vec.load('word2vec_model')
    manager('./data/traindata.txt', './data/testdata.txt')

