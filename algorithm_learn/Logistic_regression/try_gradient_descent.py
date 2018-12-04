import numpy
import random
import matplotlib.pyplot as plt
# 自己写的第一个梯度下降
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([float(lineArr[0]),float(lineArr[1]),1])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
# ????另一种思路
def loadDataSet1():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([float(lineArr[0]),1])
        labelMat.append(float(lineArr[1]))
    return dataMat,labelMat
# 梯度下降
def gradAscent(dataMatIn,classLabels):
    # mat 创建矩阵
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    rate=0.1
    row,line=numpy.shape(dataMatrix)
    # 迭代次数
    maxCycles =300
    Wk = numpy.ones((row, 1))
    Wb = numpy.ones((row, 1))
    for i in range(maxCycles):
        y=numpy.multiply(Wk, dataMatrix[:, 0]) + Wb
        Wb = Wb - (y- dataMatrix[:, 1])*rate
        Wk=Wk-numpy.multiply(numpy.multiply(Wk,dataMatrix[:,0])+Wb-dataMatrix[:,1],dataMatrix[:,0])*rate
        k=numpy.sum(Wk,0)/row
        b=numpy.sum(Wb,0)/row
        print(k,b)
    return  k,b
# 使用矩阵思路梯度下降
def gradAscent1(dataMatIn,classLabels):
    # mat 创建矩阵
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    rate=0.001
    row,line=numpy.shape(dataMatrix)
    # 迭代次数
    maxCycles =500
    W = numpy.ones((line, 1))
    for i in range(maxCycles):
        y=dataMatrix*W
        print(y)
        # 举证乘法
        W = W - dataMatrix.transpose()*(y- labelMat)*rate
        print(W)
    return  W[0,0],W[0,1]
# 随机梯度下降
def stocGradAscent0(dataMatIn,classLabels):
    # 创建矩阵
    dataMatrix=numpy.mat(dataMatIn)
    row,line=numpy.shape(dataMatrix)
    rate = 0.1
    Wk = 1
    Wb = 1
    for i in range(row):
        y=Wk*dataMatrix[i,0]+Wb
        Wb=Wb-rate*(y-dataMatrix[i,1])
        Wk=Wk-rate*(y-dataMatrix[i,1])*dataMatrix[i,0]
        print(Wk,Wb)
    return Wk,Wb
# 随机梯度下降 改进
def stocGradAscent01(dataMatIn,numIter=500):
    numIter = 500
    # 创建矩阵
    dataMatrix=numpy.mat(dataMatIn)
    row,line=numpy.shape(dataMatrix)
    rate = 0.1
    Wk = 1
    Wb = 1
    for j in range(numIter):
        dataIndex=list(range(row))
        for i in range(row):
            rate = 4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            y=Wk*dataMatrix[randIndex,0]+Wb
            Wb=Wb-rate*(y-dataMatrix[randIndex,1])
            Wk=Wk-rate*(y-dataMatrix[randIndex,1])*dataMatrix[randIndex,0]
            del (dataIndex[randIndex])
            print(Wk,Wb)
    return Wk,Wb

# 绘图
def plotPicture():
    dataMat, labelMat = loadDataSet()
    # dataMat1, labelMat1 = loadDataSet1()
    dataArr = numpy.array(dataMat)
    n = numpy.shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='red', marker='s')
    # ax.scatter(xcord2, ycord2, s=30, c='green')
    # ++++++++++++++++++++++++++++++++++
    k, b = stocGradAscent0(dataArr, labelMat)
    # k1, b1 = stocGradAscent0(dataMat1, labelMat1)
    # k, b = gradAscent(dataArr, labelMat)
    x = numpy.arange(-3.0, 3.0, 0.1)
    # print(k1,b1)
    # y = k[0,0] * x + b[0,0]
    y = k * x + b
    # y1 = k1 * x + b1
    ax.plot(x, y)  # 蓝色
    # ax.plot(x, y1)  # 蓝色
    plt.show()
# dataArr,labelMat=loadDataSet()
plotPicture()