# coding=utf-8
from snownlp import SnowNLP
import jieba
jieba.load_userdict("dict.txt")
rawstr='衣家系几点'
s=jieba.cut(str(rawstr), cut_all=False)
for i in s:
    print(i)

# s = SnowNLP(u'这个东西真心很赞这个东西真心很赞')
# print(s.tf)
# s1=SnowNLP([s.words])
#
# print(s1.tf)
# print(s1.idf)