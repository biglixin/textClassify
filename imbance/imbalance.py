# coding=utf-8
import util.preprocess as pr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

def getData(data,positiveNum,negativeNum):
    x_data=[]
    y_data=[]
    i=0
    j=0
    for d in data:
        ds =d.split('\t')
        if ds[0]=='1':
            if i > positiveNum:
                continue
            x_data.append(ds[1])
            y_data.append(ds[0])
            i = i + 1
        if ds[0]=='0':
            if j >negativeNum:
                continue
            x_data.append(ds[1])
            y_data.append(ds[0])
            j = j + 1
    return x_data,y_data

def manger():
    trainData = pr.readData('D:/Users/lianglx5/PycharmProjects/textClassfiy/data/traindata.txt')
    testData = pr.readData('D:/Users/lianglx5/PycharmProjects/textClassfiy/data/testdata.txt')
    x_train,y_train=getData(trainData,1000,10000)
    x_test,y_test=getData(testData,400,400)
    pipe = make_pipeline_imb(TfidfVectorizer(),
                             RandomUnderSampler(),
                             MultinomialNB())
    pipe.fit(x_train, y_train)
    y_test=np.array(y_test)
    y_pred = pipe.predict(x_test)
    print(classification_report_imbalanced(y_test, y_pred))

if __name__ == '__main__':
    manger()