from numpy import *
import feedparser
import os
# 创建实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性文字  0代表正常言论
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
# 把训练集的词添加到一个列表中 包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet=vocabSet|set(document)
    return list(vocabSet)
# 输入参数为词汇表及某个文档 ，输出的是文档向量，向量的每一元素为1或0，分别表示词汇表中的单词在输人文档中是否出现
# 词集模式
def setOfWords2Vec(vocabList,inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word:%s is not in my Vocabulary!" %word)
    return returnVec
# 词袋模式遇到对应值+1
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:

        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec
# 朴素贝叶斯分类器训练函数
# trainMatrix 训练集合  trainCategory 结果集合
def trainNB0(trainMatrix,trainCategory):
    # 行个数
    numTrainDocs=len(trainMatrix)
    # 列个数
    numWords=len(trainMatrix[0])
    # 总的概率
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    # 初始化概率
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom =2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    #  每个元素是侮辱词的概率
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
# 判断类别
# 判断是否是侮辱性词汇
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
#     便利函数
def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))
#   正则提取多个字符并返回列表 列表中字符均为小写
def textParse(bigString):
    import re
    # 匹配多个包括下滑线在内的任何字符
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        wordList=textParse(open(os.getcwd()+'/email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' %i,errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 把多个列表去重合成一个
    vocabList=createVocabList(docList)
    traningSet=list(range(50))
    testSet=[]
    # 随机取出10个作为测试集 剩下的作为训练集（交叉验证）
    for i in range(10):
        # 随机返回一个0，50的数
        randIndex=int(random.uniform(0,len(traningSet)))
        testSet.append(traningSet[randIndex])
        del (traningSet[randIndex])
    # 训练数据集合
    trainMat=[]
    # 训练结果集合
    trainClasses=[]
    for docIndex in traningSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
            print(errorCount)
    print('the error rate is ',float(errorCount)/len(testSet))
#     提取高频词
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]
# 由于问题暂时获取不到RSS源  RSS源分类器
def localWords(feed1,feed0):
    import feedparser
    docList=[]
    classList=[];fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet=range(2*minLen)
    testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print("the error rate is :",float(errorCount)/len(testSet))
    return vocabList,p0V,p1V
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0:
            topNY.append((vocabList[i],p1V[i]))
        sortedSF=sorted(topSF,key=lambda pair:pair[1] ,reverse=True)
        print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
        for item in sortedSF:
            print(item[0])
        sortedNY =sorted(topNY,key=lambda pair:pair[1],reverse=True)
        print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
        for item in sortedNY:
            print(item)


if __name__=="__main__":
    # listOPosts,listClasses=loadDataSet()
    # myVocabList=createVocabList(listOPosts)
    # # vector=setOfWords2Vec(myVocabList,listOPosts[0])
    # # print(vector)
    # trainMat=[]
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    # print(trainMat)
    # p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    # print(p0V)
    # print(p1V)
    # ***************************************
    # testingNB()
    # *****************************************
    # spamTest()
    # *****************************************
    # ny=feedparser.parse('http://www.runoob.com/python/func-number-uniform.html')
    # print(ny)
    ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # covabList,pSF,pNY=localWords(ny,sf)
    # vocabList,pSF,pNY=localWords(ny,sf)
    getTopWords(ny,sf)
