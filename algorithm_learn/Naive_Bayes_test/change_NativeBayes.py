import time
# 可以训练多个模型，加载多个模型到一个里面
class NaiveBayesian:
     def __init__(self,ratio = 0.00001):
         self.ratio = ratio
         self.line_num = 0  # 训练集总数
         self.type_num = {}  # 记录各评分个数
         self.type_tags = {}  # 记录各评分的属性
         self.type_probability = {}  # 类别的概率
         self.type_tag_probability = {}  # 类别下单词的概率
     def train(self,trianFile = ''):
         for i,line in enumerate(open(trianFile,'r',encoding='utf-8')):
             self.line_num += 1
             newid,type,tags = line.strip().split('\t')
             # print (i)
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
         self.get_tagnumber()
     def get_tagnumber(self):
         for type in self.type_num:
             self.type_probability[type] = float(self.type_num[type]) / float(self.line_num)
             self.type_tag_probability[type] = {}
             for tag in self.type_tags[type]:
                 self.type_tag_probability[type][tag] = float(self.type_tags[type][tag]) / float(self.type_num[type])
     def SaveModel(self,modelFile = ''):
         outFile = open(modelFile, 'w',encoding='utf-8')
         for type in self.type_num:
             outFile.write(type + ':'+str(self.type_num[type])+ ';')
             wordProbability = ''
             for tag in self.type_tag_probability[type]:
                 if wordProbability == '':
                    wordProbability = tag + ':' + str(self.type_tags[type][tag])
                 else:
                    wordProbability += ',' + tag + ':' + str(self.type_tags[type][tag])
             outFile.write(wordProbability+';'+str(self.line_num) + '\n')
         outFile.close()
     def LoadModel(self,modelfile = ''):
        file = open(modelfile,'r',encoding='utf8')
        for line in file:
            datas = line.strip('\n').split(';')
            self.line_num=datas[2]
            types=datas[0].split(':')
            self.type_num[types[0]]=types[1]
            self.type_tags[types[0]] = {}
            typeWordsDatas = datas[1].split(',')
            for typeWords in typeWordsDatas:
                words = typeWords.split(':')
                self.type_tags[types[0]][words[0]] = float(words[1])
        self.get_tagnumber()
     #  朴素贝叶斯的数学实现
     def Predict(self,tags= ''):
         tags = tags.split()
         maxValue = 0.0
         result = ''
         # 每个分类的概率应该是相同的（每次训练集中的类别应该均衡）
         for type in self.type_probability:
            # file_error0=open('error000.txt','w+')
            # predictedValue = self.type_probability[type]
            # print(predictedValue)
            predictedValue = 1
            str_test=type + ','+str(self.type_probability[type])+','
            for tag in tags:
                 if tag in self.type_tag_probability[type]:
                    str_test+=tag+','+str(self.type_tag_probability[type][tag])+','
                    predictedValue *= self.type_tag_probability[type][tag]
                 else:
                    predictedValue *= 0.00001
            predictedValue=predictedValue/self.type_probability[type]
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
             newid,type,tags = line.strip().split('\t')
             newType,str_test= self.Predict(tags)
             total_file.write(str(newid)+'\t'+str(type) + '\t'+ str(newType) + '\t'+ tags+ '\n')
             if newType == type:
                 correctNum += 1
             else:
                 error_file.write(str(newid)+'\t'+str(type) +'\t'+ str(newType) +'\t'+tags+'\n')
         error_file.close()
         return float(correctNum) / num
     def ChangeCat(self,newType):
         if newType=='彩票':
             pass
         pass

t1=time.time()
naiveBayesian = NaiveBayesian()
# naiveBayesian.train('Train_data01_34wan_clear.txt')
# naiveBayesian.train('train_data_48.txt')
# naiveBayesian.SaveModel('NaiveBayesianModel2018426_echo3')
naiveBayesian.LoadModel('NaiveBayesianMode20Typeecho3')
# print(naiveBayesian.type_tag_probability)
# pro=naiveBayesian.TestFile('train_data_48.txt')
# naiveBayesian.train('train.txt')
# naiveBayesian.SaveModel('NaiveBayesian_model_48_calss_test')
# naiveBayesian.LoadModel('NaiveBayesian_model_48_calss_test')
pro=naiveBayesian.TestFile('train_data_48.txt')
print  ('预测准确率%s' %pro)
t2=time.time()
print ('共用时 %.4f s'%(t2-t1))











