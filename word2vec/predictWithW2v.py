# coding=utf-8
from gensim.models import word2vec
import util.preprocess as pr

stopword=["我","你","係","都","會","佢","其","實"]
def predict(text):
    li = []
    for i in text:
        try:
            a=model[i]
            if i not in stopword:
                li.append(i)
        except:
            pass
    if len(li)<1:
        return 0
    y=model.n_similarity(li, dic)
    return y

if __name__ == '__main__':
    model = word2vec.Word2Vec.load('word2vec_model')
    if "." in model:
        print(1)
    else:
        print(0)
    print(model["."])
    # dic = pr.readData('dic.txt')
    # trainData = pr.readData("D:\\Users\\lianglx5\\PycharmProjects\\textClassify\\data\\alldata.txt")
    # i=0
    # for l in trainData:
    #     d=l.split('\t')
    #     if len(d)==2 and d[0]=="0" and predict(d[1])<0.5:
    #         i=i+1
    #         print(predict(d[1]),d[0],d[1])
    # print(i)