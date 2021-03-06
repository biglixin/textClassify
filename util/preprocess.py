# coding=utf-8
import jieba
import math

def performance(labelArr, predictArr):#类标签为int类型
    #labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.; TN = 0.; FP = 0.; FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == '1' and predictArr[i] == '1':
            TP += 1.
        if labelArr[i] == '1' and predictArr[i] == '0':
            FN += 1.
        if labelArr[i] == '0' and predictArr[i] == '1':
            FP += 1.
        if labelArr[i] == '0' and predictArr[i] == '0':
            TN += 1.
    POSP = TP/(TP + FP)
    POSN = TP / (TP + FN)
    POSF1= 2*POSP*POSN/(POSP+POSN)

    NEGP = TN/(FN + TN)
    NEGN = TN/(FP + TN)
    NEGF1=2*NEGP*NEGN/(NEGP+NEGN)

    print("class"+"\t"+"precision"+"\t"+"recall"+"\t"+"f1-score")
    print("1"+"\t"+str(POSP)+"\t"+str(POSN)+"\t"+str(POSF1))
    print("0"+"\t"+str(NEGP)+"\t"+str(NEGN)+"\t"+str(NEGF1))




def splitSentienceToChar(inPath,outPath):
    l=[]
    for line in readData(inPath):
        s=" ".join(line.replace("\t",""))
        l.append(s)
    saveData(outPath,l)


def getStopWordsPattern(path):
    s=""
    li=[]
    with open(path, "r",encoding="utf-8") as f:
        for line in f:
            tmp = line.strip()
            if len(tmp) == 0:
                continue
            li.append(tmp)
    s="|".join(li)
    return s
def jiebaCutWord(rawstr):
    return [i for i in jieba.cut(str(rawstr), cut_all=False)]
def saveData(path,data):
    with open(path,"w",encoding='utf-8') as f:
        for i in data:
            print(i)
            f.write(i+"\n")
def readData(path):
    li=[]
    with open(path, "r",encoding='utf-8') as f:
        for i in f:
            li.append(i.strip())
    return li
def readDataAndCutWord(path):
    li=[]
    with open(path, "r",encoding='utf-8') as f:
        for i in f:
            l=jiebaCutWord(i.strip())
            li.append(i.strip())
    return li

def spiltData(path,num):
    train_toall=[]
    test_toall=[]
    positive=[]
    negtive=[]
    with open(path,"r",encoding='utf-8') as f:
        for line in f:
            s =line.strip()
            print(s[0])
            if s[0]=="1":
                positive.append(s)
            elif s[0]=="0":
                negtive.append(s)
    train_toall.extend(positive[0:int(len(positive)*num)])
    test_toall.extend(positive[int(len(positive)*num):])
    train_toall.extend(negtive[0:int(len(negtive)*num)])
    test_toall.extend(negtive[int(len(negtive)*num):])
    saveData("./data/traindata.txt",train_toall)
    saveData("./data/testdata.txt",test_toall)

if __name__ == '__main__':
    # spiltData("./data/11.txt",0.8)
    splitSentienceToChar("D:/Users/lianglx5/PycharmProjects/textClassfiy/data/11.txt","D:/Users/lianglx5/PycharmProjects/textClassfiy/data/33.txt")