# coding=utf-8
# coding=utf-8
from filecmp import cmp
import util.preprocess as pr
import pandas as pd
import sys
import re
T=""
from pandas.core import index


def ngramHelper(datas, num, results):
    for i in num:
        for data in datas:
            # data=data.split("[.!?]+")
            data = re.split('[^\u4E00-\u9FA5]', str(data))
            for d in data:
                for word in ngramExact(d.strip(), i):
                    if len(word) < 1:
                        continue
                    if word in results[i]:
                        results[i][word] = results[i][word] + 1
                    else:
                        results[i][word] = 1
        results[i] = sorted(results[i].items(), key=lambda d: d[1], reverse=True)


def ngramExact(rawstr, num):
    splits = [i for i in rawstr]
    result = []
    num = int(num)
    if num == 1:
        result.extend(splits)
    elif num == 2 and len(splits) >= 2:
        resulttmp = [splits[i] + T + splits[i + 1] for i in range(0, len(splits) - 1)]
        result.extend(resulttmp)
    elif num == 3 and len(splits) >= 3:
        resulttmp = [splits[i] + T + splits[i + 1] + T + splits[i + 2] for i in range(0, len(splits) - 2)]
        result.extend(resulttmp)
    elif num == 4 and len(splits) >= 4:
        resulttmp = [splits[i] + T + splits[i + 1] + T + splits[i + 2] + T + splits[i + 3] for i in
                     range(0, len(splits) - 3)]
        result.extend(resulttmp)
    elif num == 5 and len(splits) >= 5:
        resulttmp = [splits[i] + T + splits[i + 1] + T + splits[i + 2] + T + splits[i + 3] + T + splits[i + 4]
                     for i in range(0, len(splits) - 4)]
        result.extend(resulttmp)
    elif num == 6 and len(splits) >= 6:
        resulttmp = [
            splits[i] + T + splits[i + 1] + T + splits[i + 2] + T + splits[i + 3] + T + splits[i + 4] + T +
            splits[i + 5] for i in range(0, len(splits) - 5)]
        result.extend(resulttmp)
    return result


def readExcel(path):
    l = []
    data = pd.read_excel(path)
    try:
        textdata = data["Comments/Replies"]
        s = textdata.index
        textdata1 = data["TopComment"]

        for i in textdata.index.values:
            if str(textdata[i]).find("+1") != -1:
                l.append("0" + "\t" + str(textdata1[i]))
            else:
                l.append("0" + "\t" + str(textdata[i]))
    except:
        print(u"大哥原文件[内容]这一列找不到")
        sys.exit()
    return l


if __name__ == '__main__':
    results = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}, "6": {}}
    testData = readExcel("D:/Users/lianglx5/Desktop/粤语/22.xlsx")
    # testData = pr.readData("D:/Users/lianglx5/Desktop/粤语/risk.txt")
    ngramHelper(testData,"2",results)
    with open('D:/Users/lianglx5/Desktop/粤语/risk_ngram_tf.txt', 'w', encoding="utf-8") as f:
        for k, v in results.items():
            for k1 in v:
                print(k, k1[0], k1[1])
                s = k + '\t' + k1[0] + '\t' + str(k1[1])
                f.write(s + '\n')
    print(u"处理结束.......")


