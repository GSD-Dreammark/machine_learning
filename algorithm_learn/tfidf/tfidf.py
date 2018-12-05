#!/usr/bin/env python
#coding=utf-8
import math
import time
# codecs专门用作编码转换
import codecs
#  operator 高级运算符包
from operator import itemgetter
class TfIdf:
  #   corpus_filename 语料库名字  stopword_filename 停用库名字 DEFAULT_IDF 默认逆文本频率指数
  def __init__(self, corpus_filename=None, stopword_filename=None,
               DEFAULT_IDF=1.5):
      self.num_docs = 0      # 文件的行数  即样本数
      self.term_num_docs = {}  # term : num_docs_containing_term  文件中出现的词及次数
      self.stopwords = set([])
      self.idf_default = DEFAULT_IDF
      if corpus_filename:
          self.merge_corpus_document(corpus_filename)
      if stopword_filename:
          stopword_file = codecs.open(stopword_filename, "r", encoding='utf-8')
          self.stopwords = set([line.strip() for line in stopword_file])
  # 加载语料库
  def merge_corpus_document(self, corpus_filename):
          """slurp in a corpus document, adding it to the existing corpus model
          """
          corpus_file = codecs.open(corpus_filename, "r", encoding='utf-8')
          # Load number of documents.
          # 第一行表示样本个数
          line = corpus_file.readline()
          self.num_docs += int(line.strip())
          # Reads "term:frequency" from each subsequent line in the file.
          for line in corpus_file:
              tokens = line.rsplit(":", 1)
              term = tokens[0].strip()
              try:
                  frequency = int(tokens[1].strip())
              except Exception as err:
                  if line in ("", "\t"):
                      # catch blank lines
                      print("line is blank")
                      continue
                  else:
                      raise
              if self.term_num_docs.has_key(term):
                  self.term_num_docs[term] += frequency
              else:
                  self.term_num_docs[term] = frequency
  # 训练模型
  def train_model(self, path):
    """Add terms in the specified document to the idf dictionary."""
    num=0
    self.term_num_docs={}
    for line in open(path,'r',encoding='utf-8'):
      num+=1
      line0=line.split('\n')
      # print(line0[0])
      lines=line0[0].split(' ')
      # 去除本次样本的重复数据
      lines=set(lines)
      # 计算出现的词及出现次数
      for word in lines:
        if word in self.term_num_docs:
          self.term_num_docs[word] += 1
        else:
          self.term_num_docs[word] = 1
    self.num_docs=num
  #    保存训练模型
  def save_corpus_to_file(self, idf_filename):
    """Save the idf dictionary and stopword list to the specified file."""
    output_file = codecs.open(idf_filename, "w", encoding='utf-8')
    output_file.write(str(self.num_docs) + "\n")
    for term, num_docs in self.term_num_docs.items():
       output_file.write(term + ": " + str(num_docs) + "\n")
  # 获取样本数
  def get_num_docs(self):
    """Return the total number of documents in the IDF corpus."""
    return self.num_docs
  def get_idf(self, term):
    if not term in self.term_num_docs:
      return self.idf_default
    # 分子分母都+1 取对数防止分母为0  ，但本代码做了处理故不需要+1
    return math.log(float(self.get_num_docs()) / (self.term_num_docs[term]))
    # return math.log(float(1 + self.get_num_docs()) /(1 + self.term_num_docs[term]))
  def word_tfidf(self, curr_doc,out_filename,num):
    f = open(out_filename, 'w')
    for line in open(curr_doc,'r',encoding='utf-8'):
        lis0=[]
        lines=line.split()
        print(lines)
        if len(lines)>20 :
         tfidf = {}
         for word in lines:
             # 求出word 有多少个
             mytf = float(line.count(word)) / len(lines)
             myidf = self.get_idf(word)
             tfidf[word] = mytf * myidf
         word_tf=sorted(tfidf.items(), key=itemgetter(1), reverse=True)
         for i,item in enumerate(word_tf):
            print(i,item)
            if i>(num-1):
                break
            lis0.append(item[0])
         st=' '.join(lis0)
         f.write(st+'\n')
  # 获取一片文章的关键词
  def one_tfidf(self,line,num):#line输入为分词结果
      lis0=[]
      lines=line.split()
      if len(lines)>20 :
         tfidf = {}
         for word in lines:
             # line.count（word） word在这一行的个数
             mytf = float(line.count(word)) / len(lines)
             myidf = self.get_idf(word)
             tfidf[word] = mytf * myidf
         # 以字典中的value来排序  itemgetter(1)==匿名函数lambda s:s[1]
         word_tf=sorted(tfidf.items(), key=itemgetter(1), reverse=True)
         # 取前num 条数据
         for i,item in enumerate(word_tf):
            if i>(num-1):
                break
            lis0.append(item[0])
         st=' '.join(lis0)
         return st
  # 加载训练模型   （把文档中的数据写到类TfIdf 的term_num_docs）
  def loda_model(self,corpus_filename):
      # enumeration为可遍历的变量的索引
      for i,line in enumerate(open(corpus_filename,'r',encoding='utf-8')):
          if  i==0:
            self.num_docs=int(line.strip())
          else:
            lines=line.split(':')
            self.term_num_docs[lines[0]]=int(lines[1].strip())
time1=time.time()



Tf=TfIdf()
Tf.train_model('test_out19.txt')
Tf.save_corpus_to_file('corpus_.txt')
Tf.loda_model('corpus_.txt')
# Tf.word_tfidf('corpus_.txt','test.txt',20)
keyword=Tf.one_tfidf('前不久 梦想 合伙人 发布会 在京举行 这部 电影 导演 制片人 主演'
                     ' 纷纷 到场 女主有 唐嫣 姚晨 郝蕾 女神 级别 演员 小编 印象 最深'
                     ' 郝蕾 一改 往日 性感 路线 玩起 少女 诧异 郝蕾 穿着 黑色 蕾丝 '
                     '翻领 上衣 搭配 一袭 白色 高腰 长款 清爽 可爱 丸子 头立显 清纯'
                     ' 可爱 少女感 网友 直呼 邓超 后悔 简约 黑白 两色 搭配 完美 融入 蕾丝 '
                     '元素 纯白 长款 温柔 搭配 黑色 蕾丝 白色 领子 上衣 显得 郝蕾 清纯 可爱 温柔 轻薄 透亮 底妆 打造出 一种 妆感 妆容 口红 选用 显气色 夸张 橘色 整体 妆容 时装 搭配 '
                     '显得 郝蕾 清纯 高高 束起 丸子 展现 俏皮 可爱 少女气息 高腰 腰线 完美 展现出 郝蕾 细细的 杨柳 显得 腿部 修长 耳朵 旁边 乱跑 心机 小碎发 增加 整体造型 活泼 清纯 可爱 郝蕾托 莞尔一笑 粉丝 倾倒 随意 编织 '
                     '麻花辫 可爱 少女 发型 刘海 戴上 一顶 精致 小巧 帽子 展现 活泼 少女 气质 发型 搭配白色 上衣 休闲 短裤 更显 青春 活泼 气息 搭配 选用 精致 小巧 配饰 包包 简约 凉鞋 整体造型 增加 休闲 无辜'
                     '眼睛 随意 扎起来 俏皮 马尾 微微 嘟嘴 卖个 妆感 更能 展现 郝蕾 满满的 清纯 少女 活力 干净 清爽 利落 短发 搭配 香槟色 发箍 造型 展现 郝蕾 满满的 活泼 少女气息 清透 皮肤 自然 唇色 炯炯有神 眼睛 可爱 少女 必备 元素 '
                     ,20)
print (keyword)
time2=time.time()
print ('时间',time2-time1)



