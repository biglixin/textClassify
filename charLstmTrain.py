# coding=utf-8

'''
使用字向量作为lstm模型的输入，训练lstm模型
'''

import numpy as np
import pandas as pd

pos = pd.read_table('./data/traindata.txt', header=None,sep='\t')
neg = pd.read_table('./data/testdata.txt', header=None,sep='\t')
label=pos[0].append(neg[0], ignore_index=True)
all_= pos.append(neg, ignore_index=True)
all_['label']=label

# print(all_[1])
# print(all_['label'])

maxlen = 200 #截断字数
min_count = 20 #出现次数少于该值的字扔掉。这是最简单的降维方法

content = ''.join(all_[1])
abc = pd.Series(list(content)).value_counts()
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)
abc[''] = 0 #添加空字符串用来补全
word_set = set(abc.index)

def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])

all_['doc2num'] = all_[1].apply(lambda s: doc2num(s, maxlen))
print(all_['doc2num'])
#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1,1)) #调整标签形状


from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128
train_num = 20000

model.fit(x[:train_num], y[:train_num], batch_size = batch_size, nb_epoch=1)

model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)

# model.save('my_model.h5')

