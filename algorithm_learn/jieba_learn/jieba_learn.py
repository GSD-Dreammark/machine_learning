import jieba
import jieba.posseg as pseg
import jieba.analyse
# 安装jieba
# pip install jieba
str1='我来到北京清华大学'
str2='python的正则表达式是最好的'
str3='小明硕士毕业于中国科学院计算机所，所在日本京都大学深造'
# 支持三种分词模式
# 1 精确模式，试图将句子最精确地切开，适合文本分析；
# 2 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义； 适用自定义词典强大的
# 3 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
# 支持繁体分词
# 支持自定义词典
# 分词
# jieba.cut 返回的是generator jieba.lcut 返回的是list  lcut与cut参数一样都适用3种模式
seg_list=jieba.cut(str1,cut_all=True)       ##全模式 (模式1)
result = pseg.cut(str1)                     ##词性标注，标注句子分词后每个词的词性
result2 = jieba.cut(str2)                   ##默认是精准模式（模式2）
# 关键词抽取
# 参数：
# sentence ：为待提取的文本
# topK： 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
# withWeight ： 为是否一并返回关键词权重值，默认值为 False
# allowPOS ： 仅包括指定词性的词，默认值为空，即不筛选
result3 =  jieba.analyse.extract_tags(str1,2)
##关键词提取，参数setence对应str1为待提取的文本,topK对应2为返回几个TF/IDF权重最大的关键词，默认值为20
result4 = jieba.cut_for_search(str3)        ##搜索引擎模式（模式3）
print('/'.join(seg_list))
for w in result:
    print(w.word,w.flag)
for t in result2:
    print(t)
for s in result3:
    print(s)
print(','.join(result4))


#  返回词所在的位置
test_sent='永和服装饰品有限公司'
result5=jieba.tokenize(test_sent)# 返回词语在原文的起始位置
for tk in result5:
    print('word %s   start:%d   end:%d' %(tk[0],tk[1],tk[2]))
    print(tk)
# 自定义词典
import sys
jieba.load_userdict('userdict.txt')
test_send='大连美容美发学校中君意是你值得信赖的选择'
test_sent2 = '江州市长江大桥参加了长江大桥的通车仪式'
print (", ".join(jieba.cut(test_send)))
print (", ".join(jieba.cut(test_sent2)))
# 使用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其（或不能）被分出来。
jieba.suggest_freq(('中出'),True)
"""
自定义词典的格式：一个词占一行；每一行分三部分，一部分为词语，另一部分为词频，最后为词性（可省略），用空格隔开
其中user_dict.txt的内容是：
云计算 5
李小福 2 nr
创新办 3 i
easy_install 3 eng
好用 300
韩玉赏鉴 3 nz
八一双鹿 3 nz
台中
凱特琳 nz
Edu Trust认证 2000
君意 3
江大桥 200
"""
