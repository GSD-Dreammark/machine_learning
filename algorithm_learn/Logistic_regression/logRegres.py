import math
import random
# 假设回归函数为 w0*x0+w1*x1+w2*x2=0   w0=b w1=x w2=y x0=1
import numpy
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
# 映射函数 使参数映射到0，1区间
def sigmoid(inX):
    # e的-inX次方
    return 1.0/(1+numpy.exp(-inX))
# （返回最大值的梯度 ）梯度上升算法
def gradAscent(dataMatIn,classLabels):
    # mat 创建矩阵h
    dataMatrix=numpy.mat(dataMatIn)
    labelMat=numpy.mat(classLabels).transpose()
    m,n=numpy.shape(dataMatrix)
    # 步长
    alpha=0.001
    # 迭代次数
    maxCycles=500
    weights=numpy.ones((n,1))
    # print(weights)
    for k in range(maxCycles):
        # 特征*回归系数
        h=sigmoid(dataMatrix*weights)
        print(h)
        error=(labelMat-h)
        weights=weights+alpha * dataMatrix.transpose()*error
        # print(weights)
    return weights
# 回执图像
# 假设回归函数为 w0*x0+w1*x1+w2*x2=0   w0=b w1=x w2=y x0=1
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=numpy.array(dataMat)
    n=numpy.shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2 = [];ycord2 = []
    for i in range(n):
        if  int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=numpy.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*(x))/weights[2]
    ax.plot(x,y)
    plt.show()
# 随机梯度上升
def stocGradAscent0(dataMatrix,classLabels):
    m,n=numpy.shape(dataMatrix)
    alpha=0.01
    weights=numpy.ones(n)
    # 循环所有特征 获取回归系数
    for i in range(m):
        # 计算一组特征的梯度
        h=sigmoid(numpy.sum(dataMatrix[i]*weights))
        # h是一个值 ，梯度上升h是向量
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights
# 随机梯度上升 改进
def stocGradAscent1(dataMatrix,classLabels,numIter=500):
    m,n=numpy.shape(dataMatrix)
    weights=numpy.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
def classifyVector(inX,weights):
    prob=sigmoid(numpy.sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        # 将字符类型转化为float
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(numpy.array(trainingSet),trainingLabels,500)
    # trainWeights = gradAscent(numpy.array(trainingSet), trainingLabels)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(numpy.array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print('the error rate of this test is :%f' %errorRate)
    return errorRate
def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is: %f" %(numTests,errorSum/float(numTests)))



if __name__=='__main__':
    dataArr,labelMat=loadDataSet()
    weights1=gradAscent(dataArr,labelMat)
    weights1=weights1.getA()
    # #从矩阵类型变为数组类型 .getA()
    plotBestFit(weights1)
    weights2=stocGradAscent0(numpy.array(dataArr),numpy.array(labelMat))
    plotBestFit(weights2)
    weights3=stocGradAscent1(numpy.array(dataArr),labelMat)
    # plotBestFit(weights3)
    # 将三种方式的回归线放到一张图上，可以很明显的看出梯度上升算法要优与随机梯度算法
    import matplotlib.pyplot as plt

    dataMat, labelMat = loadDataSet()
    dataArr = numpy.array(dataMat)
    n = numpy.shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x1 = numpy.arange(-3.0, 3.0, 0.1)
    # 最后值取0 是因为0是sigmoid 的分界值 =0.5
    y1 = (-weights1[0] - weights1[1] * (x1)) / weights1[2]
    x2 = numpy.arange(-3.0, 3.0, 0.1)
    y2 = (-weights2[0] - weights2[1] * (x2)) / weights2[2]
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weights3[0] - weights3[1] * (x)) / weights3[2]
    ax.plot(x, y)# 蓝色
    # 梯度
    ax.plot(x1, y1)# 橙色
    # 随机梯度
    ax.plot(x2, y2)# 绿色
    plt.show()
    multiTest()