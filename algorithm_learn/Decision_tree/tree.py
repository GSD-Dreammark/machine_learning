from math import log
from Decision_tree import treePlotter
import operator
def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
#  计算给定数据集的香浓熵（信息期望值）
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        # key的概率
        prob=float(labelCounts[key]/numEntries)
        shannonEnt-=prob * log(prob,2)
    return shannonEnt
# 按照给定的特征划分数据集
# 参数：待划分的数据集、划分数据集的特征、特征的返回值
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        # 数组下标为axis的值==value
        if featVec[axis]==value:
            # 把数据的特征值从数据中除去
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList =[example[i] for example in dataSet]
        uniqueVale=set(featList)
        newEntropy=0.0
        # 求每种划分的香农熵
        for value in uniqueVale:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        # 取出最大（香农熵最小的）的并返回下标
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature
# 选取可能性最大的结果值
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    #  返回的是[(,),]
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
# 创建树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    # 类别相同
    if classList.count(classList[0]) ==len(classList):
        return classList[0]
    # 特征值为一
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels =labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
# 使用决策树的分类函数
# inputTree 输入树  featLables 特征属性标签集 testVec列表 存放的子节点
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    # 将标签字符串转化为索引
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel
# 将决策树存储到pickle模块中
def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
#     加载决策树
def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)
if __name__=="__main__":
    myDat,labels=createDataSet()
    # entropy=calcShannonEnt(myDat)
    # print(entropy)
    # # 熵越高，混合的数据也越多 （熵随着类别的增加而变大）
    # myDat[0][-1]='maybe'
    # entropy = calcShannonEnt(myDat)
    # print(entropy)
    # ************************************
    # result=splitDataSet(myDat,0,1)
    # print(result)
    # result = splitDataSet(myDat, 0, 0)
    # print(result)
    #************************************
    # Best_feature=chooseBestFeatureToSplit(myDat)
    # print(Best_feature)
    # ****************************************
    # trees=createTree(myDat,labels)
    # print(trees)
    # *****************************************
    # print(labels)
    # mytree=treePlotter.retrieveTree(0)
    # print(mytree)
    # class_label=classify(mytree,labels,[1,0])
    # print(class_label)
    # class_label = classify(mytree, labels, [1, 1])
    # print(class_label)
    # **************************************
    # mytree=treePlotter.retrieveTree(0)
    # storeTree(mytree,'classifierStorage.txt')
    # ss=grabTree('classifierStorage.txt')
    # print(ss)
    # 使用决策树判断隐形眼镜问题
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t')for inst in fr.readlines()]
    lensesLabels=['age','prescript','astigmatic','tearRate']
    lenseTree=createTree(lenses,lensesLabels)
    print(lenseTree)
    treePlotter.createPlot(lenseTree)