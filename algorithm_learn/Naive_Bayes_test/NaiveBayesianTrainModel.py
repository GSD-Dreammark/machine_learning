import time

class NaiveBayesian:
     def __init__(self,ratio = 0.00001):
         self.ratio = ratio
     def train(self,trianFile = ''):
         self.line_num = 0#训练集总数
         self.type_num = {}#记录各评分个数
         self.type_tags = {}#记录各评分的属性
         for i,line in enumerate(open(trianFile,'r',encoding='utf-8')):
             self.line_num += 1
             newid,very_old,old,type,tags = line.strip().split('\t')
             print (i)
             if type in self.type_num:
                 self.type_num[type] += 1
             else:
                 self.type_num[type] = 1
                 self.type_tags[type] = {}
             tags = tags.split()#关键词用空格分开
             for tag in tags:
                 if tag=='':
                     pass
                 else:
                   if tag in self.type_tags[type]:
                      self.type_tags[type][tag] += 1
                   else:
                      self.type_tags[type][tag] = 1

         self.type_probability = {} #类别的概率
         self.type_tag_probability = {}#类别下单词的概率
         for type in self.type_num:            
             self.type_probability[type] = float(self.type_num[type]) / self.line_num
             self.type_tag_probability[type] = {}
             for tag in self.type_tags[type]:
                 self.type_tag_probability[type][tag] = float(self.type_tags[type][tag]) / self.type_num[type]
     def SaveModel(self,modelFile = ''):
         outFile = open(modelFile, 'w',encoding='utf-8')
         for type in self.type_num:                  
             outFile.write(type + ':' + str(self.type_probability[type]) + ';')
             wordProbability = ''                          
             for tag in self.type_tag_probability[type]:
                 if wordProbability == '':
                    wordProbability = tag + ':' + str(self.type_tag_probability[type][tag])
                 else:
                    wordProbability += ',' + tag + ':' + str(self.type_tag_probability[type][tag])                 
             outFile.write(wordProbability + '\n')
         outFile.close()
     def LoadModel(self,modelfile = ''):
        self.type_probability = {}
        self.type_tag_probability = {}
        file = open(modelfile,'r',encoding='utf8')
        for line in file:
            datas = line.strip('\n').split(';')
            typeDatas = datas[0].split(':')
            self.type_probability[typeDatas[0]] = float(typeDatas[1])
            self.type_tag_probability[typeDatas[0]] = {}
            typeWordsDatas = datas[1].split(',')
            for typeWords in typeWordsDatas:
                words = typeWords.split(':')
                self.type_tag_probability[typeDatas[0]][words[0]] = float(words[1])
     #  朴素贝叶斯的数学实现  返回一个概率最大的类别（类别标签的概率*该类别下tag的概率）
     def Predict(self,tags= ''):
         tags = tags.split()
         maxValue = 0.0
         result = ''
         for type in self.type_probability:
            # file_error0=open('error000.txt','w+')
            predictedValue = self.type_probability[type]
            str_test=type + ','+str(self.type_probability[type])+','
            for tag in tags:
                 if tag in self.type_tag_probability[type]:
                    str_test+=tag+','+str(self.type_tag_probability[type][tag])+','
                    predictedValue *= self.type_tag_probability[type][tag]
                 else:
                    predictedValue *= 0.00001
            if maxValue < predictedValue:
                maxValue = predictedValue
                result = type
            str_test += str(predictedValue)
         return result,str_test
     def TestFile(self,testFile = ''):
         total_file=open('ClassesTrainTotal3.txt','w',encoding='utf-8')
         error_file=open('ClassesTrainError3.txt', 'w+',encoding='utf-8')
         num = 0
         correctNum = 0
         for line in open(testFile,'r',encoding='utf-8'):
             num += 1
             newid,very_old,old,type,tags = line.strip().split('\t')
             newType,str_test= self.Predict(tags)
             total_file.write(str(newid)+'\t'+very_old+'\t'+old+'\t'+str(type) + '\t'+ str(newType) + '\t'+ tags+ '\n')
             if newType == type:
                 correctNum += 1
             else:
                 error_file.write(str(newid)+'\t'+very_old+'\t'++old+'\t'+str(type) +'\t'+ str(newType) +'\t'+tags+'\n')
         error_file.close()
         return float(correctNum) / num
     def ChangeCat(self,newType):
         if newType=='彩票':
             pass


         pass

t1=time.time()
naiveBayesian = NaiveBayesian()
naiveBayesian.train('Train_data01_34wan_clear.txt')
naiveBayesian.SaveModel('NaiveBayesianModel2018426_echo2')
naiveBayesian.LoadModel('NaiveBayesianModel2018426_echo2')
pro=naiveBayesian.TestFile('ClassesTrainTotal2.txt')

# naiveBayesian.train('train.txt')5
# naiveBayesian.SaveModel('NaiveBayesian_model_48_calss_test')
# naiveBayesian.LoadModel('NaiveBayesian_model_48_calss_test')
# pro=naiveBayesian.TestFile('test.txt')
print  ('预测准确率%s' %pro)
t2=time.time()
print ('共用时 %.4f s'%(t2-t1))











