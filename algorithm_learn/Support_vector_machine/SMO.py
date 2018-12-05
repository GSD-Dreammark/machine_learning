# Platt 的 SMO 算法
# 简易版
import random
import pandas
import numpy
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
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
# 简化版的SMO算法
# dataMatIn数据集，classLabels类别标签，C常数C toler容错率 maxIter取消前最大的循环次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=numpy.mat(dataMatIn);labelMat=numpy.mat(classLabels).transpose()
    b=0;m,n=numpy.shape(dataMatrix)
    # 创建一个alpha向量并将其初始化为O向量
    alphas=numpy.mat(numpy.zeros((m,1)))
    iter=0
    # 当迭代次数小于最大迭代次数
    while (iter<maxIter):
        alphaPairsChanged=0
        # 对数据集中的每个数据向
        for i in range(m):
            # fxi 预测的类别
            fxi=float(numpy.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            # Ei 误差
            Ei=fxi-float(labelMat[i])
            # 如果该数据向量可以被优化
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                # 随机选择另外一个数据向量
                j=selectJrand(i,m)
                # 同时优化这两个向量
                fxj=float(numpy.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fxj-float(labelMat[j])
                # 为plphaIold 重新分配地址
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                # 使alpha 在范围内
                if (labelMat[i] != labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+ alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果两个向量都不能被优化，退出内循环
                if L==H:
                    print("L==H")
                    continue
                # eta 是alpha[j]的最优修改量（公式分母的部分 2k12-k11-k22）
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >=0:
                    print("eta>=0")
                    continue
                #     j 的修改(同公式一致)
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                # 使其更新后的值满足约束
                alphas[j]=clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print("j not moving enough")
                    continue
                #  对i 进行修改 ，修改量与j 相同（j 增加或减少多少 i 按同样的方式处理） ，但方向相反 此处方向相反的处理不是很理解？？？？
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                # if (alphaJold-alphas[j])>0:
                #     alphas[i]-=alphaJold-alphas[j]
                # else:
                #     alphas[i]+=alphaJold-alphas[j]
                # b 的迭代公式
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif (0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter:%d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged))

        # 如果所有向量都没被优化，增加迭代数目，继续下一次循环
        if (alphaPairsChanged==0):
            iter+=1
        else:
            iter=0
        print('iteration number:%d'%iter)
    return b,alphas
if __name__ == '__main__':
    dataArr,labelArr=loadDataSet('testSet.txt')
    b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
    # print(labelArr)
    # 是数组过滤（arrayfiltering) 的一个实例，而且它只对NumPy类型有用
    print(alphas[alphas>0])
    for i in range(100):
        if alphas[i]>0:
            print(dataArr[i],labelArr[i])



