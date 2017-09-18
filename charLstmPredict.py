# coding=utf-8
from keras.models import load_model
import numpy as np
model = load_model('my_model.h5')

def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])

def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(s, maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]

predict_one("本想着来看韩国哥的没想到是一群逗比")