import numpy as np
import time

def loadData(fileName):

    dataArr = []
    labelArr = []

    fr = open(fileName)
    i = 0
    for line in fr.readlines():
        if i == 100:
            break
        curline = line.strip().split(',')
        dataArr.append([int(int(num)>128) for num in curline[1:]])
        labelArr.append(int(curline[0]))
        i += 1

    return dataArr, labelArr

# P(y|x) = P(x|y)*P(y)/P(x)
# P(y)是先验概率
def getAllProbability(trainDataArr, trainLabelArr):
    '''
    根据训练数据，计算出先验概率和条件概率
    '''
    featureNum = 784
    classNum = 10

    # Py是先验概率分布，通过极大似然估计
    Py = np.zeros((classNum, 1))
    for i in range(classNum):
        # 计算出label==i的个数，分子+1是为了防止有的label一次都没出现，分母加K，是为了保持分子和等于分母，归一化。
        Py[i] = ((np.sum(np.mat(trainLabelArr)==i))+1)/(len(trainLabelArr)+10)
    # 取对数，这里有60000条数据，很容易产生数据下溢，尤其是后面相乘的时候，所以取对数
    Py = np.log(Py)

    # 条件概率 P(x|y)
    Px_y = np.zeros((classNum, featureNum, 2))
    for i in range(len(trainDataArr)):

        # 第i条数据的label
        label = trainLabelArr[i]
        # 第i条数据的data
        x = trainDataArr[i]
        for j in range(featureNum):
            # x[j]代表第j个特征的值 Px_y[label][j][x[j]]表示label标签第j个特征值中0和1的数量
            Px_y[label][j][x[j]] += 1


    for label in range(classNum):
        #循环每一个标记对应的每一个特征
        for j in range(featureNum):
            #获取y=label，第j个特征为0的个数
            Px_y0 = Px_y[label][j][0]
            #获取y=label，第j个特征为1的个数
            Px_y1 = Px_y[label][j][1]
            #对式4.10的分子和分母进行相除，再除之前依据贝叶斯估计，分母需要加上2（为每个特征可取值个数）
            #分别计算对于y= label，x第j个特征为0和1的条件概率分布
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    #返回先验概率分布和条件概率分布
    return Py, Px_y


def NavieBayes(Py, Px_y, x):
    '''
    Py: 先验概率
    Px_y: 条件概率
    x: 估计样本x
    '''

    featureNum = 784
    classNum = 10
    P = [0] * classNum

    for i in range(classNum):
        sum = 0
        for j in range(featureNum):
            sum += Px_y[i][j][x[j]]
        P[i] = sum + Py[i]

    return P.index(max(P))

trainDataArr, trainLabelArr = loadData('./Mnist/mnist_train.csv')
Py, Px_y = getAllProbability(trainDataArr, trainLabelArr)