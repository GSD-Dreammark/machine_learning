# 完整版
from numpy import *
import os
class optStruct:
    # dataMatIn数据集，classLabels类别标签，C常数C toler容错率
    def __init__(self,dataMatIn,classLabels,C,toler,kTup=['lin']):
        self.X=dataMatIn
        print(self.X[0,:])
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        # 误差缓存
        self.eCache=mat(zeros((self.m,2)))
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j
# 推测 H是最大值 L是最小值 ，把一个值约束在一个范围内
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L >aj:
        aj=L
    return aj
# Ei 的误差
def calcEk(oS,k):
    # 非核函数版
    # fXk=float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    # 核函数版
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return Ek
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
# 选择J 及J误差 Ej
def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    # 获取非0 元素个数
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    # 第一次选择时与简易SMO算法一致 之后 选择 i 相关的（通过 nonzero 选取非0 行） 选择差值中最大的一个（不包括本身）
    if (len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej
#  计算误差值并存人缓存
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]
#   更新alpha 和 b
def innerL(i,oS):
    Ei=calcEk(oS,i)
    if ((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphIold=copy(oS.alphas[i]);alphJold=copy(oS.alphas[j])
        if (oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.alphas[j]-oS.alphas[i]+oS.C)
        else:
            L=max(0,oS.alphas[i]+oS.alphas[j]-oS.C)
            H=min(oS.C,oS.alphas[i]+oS.alphas[j])
        if L==H:
            print("L==H")
            return 0
        # 非核函数版
        # eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        # 核函数版
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta>=0:
            print("eta>=0")
            return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if (abs(oS.alphas[j]-alphJold)<0.00001):
            print("j not miving enough ")
            return 0
        oS.alphas[i]+=oS.labelMat[i]*oS.labelMat[j]*(alphJold-oS.alphas[j])
        updateEk(oS,i)
        # 核函数版
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphIold)*oS.K[i,i]-oS.labelMat[j]*(oS.alphas[j]-alphJold)*oS.K[i,j]
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphJold)*oS.K[j,j]
        # 非核函数版
        # b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphJold)*oS.X[i,:]*oS.X[j,:].T
        # b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphJold)*oS.X[j,:]*oS.X[j,:].T
        if (0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
            oS.b=b1
        elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0

# 控制循环
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while(iter < maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            # 遍历任意可能的alpha值
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            print("fullSet,iter:%d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged))
            iter +=1
        else:
            # 遍历所有非边界的alpha值
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print("non-bound,iter:%d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif (alphaPairsChanged==0):
            entireSet=True
        print("iteration number:%d"% iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr);
    labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def testRbf(k1=1.3):
    dataArr,labelArr=loadDataSet('testSetRBF.txt')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    datMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    # list 不支持 仅在numpy 中支持 这种遍历方式
    sVs=datMat[svInd]
    labelSV=labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n=shape(datMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,datMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        # sign 对于非复数 是用来判断是-1 还是 +1 的  ==x/abs(x)
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1
        print("the training error rate is: %f" %(float(errorCount)/m))
        dataArr,labelArr=loadDataSet('testSetRBF2.txt')
        errorCount=0
        datMat=mat(dataArr);labelMat=mat(labelArr).transpose()
        m,n=shape(datMat)
        for i in range(m):
            kernelEval=kernelTrans(sVs,datMat[i,:],('rbf',k1))
            predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
            if sign(predict)!=sign(labelArr[i]):
                errorCount+=1
        print("the test error rate is:%f" %(float(errorCount)/m))


# 手写字优化  此处只识别1,9 这个二分类问题
#  将文本读入到数组中，将32*32存为1*1024
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        # 取出前32个字符存入到数组中
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
def loadImages(dirName):
    from os import listdir
    hwLabels=[]
    trainingFileList=listdir(dirName)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr==9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:]=img2vector('%s/%s' %(dirName,fileNameStr))
    return trainingMat,hwLabels
def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImages('D:/www/algorithm_learn/knn/trainingDigits')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    datMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV=labelMat[svInd]
    # 支持向量个数就是行数
    print("there are %d Support Vectors" %shape(sVs)[0])
    m,n=shape(datMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1
    print("the training error rate is :%f" %(float(errorCount)/m))
    dataArr,labelArr=loadImages('D:/www/algorithm_learn/knn/testDigits')
    errorCount=0
    datMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    m,n=shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :],kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is : %f" %(float(errorCount)/m))

if __name__ == '__main__':
    # dataArr,labelArr=loadDataSet('testSet.txt')
    # b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
    # ws=calcWs(alphas,dataArr,labelArr)
    # print(ws)
    # dataMat=mat(dataArr)
    # print(dataMat[0]*mat(ws)+b)
    # print(dataMat[2]*mat(ws)+b)
    # print(dataMat[1]*mat(ws)+b)
    # testRbf()
    testDigits(kTup=('rbf', 100))

