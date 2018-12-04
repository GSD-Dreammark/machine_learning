import re
import jieba
import pandas as pd
import random
import time
import numpy as np
import logger
import os
import tfidf
from Mysql_Class import MSSQL
logger=logger.get_logger(os.getcwd()+'/log/O_log.log')
#regular
def Regular(line):
    dr = re.compile(r'<[^>]+>', re.S)
    dd = dr.sub('', line)
    dd2 = ''.join(dd.split())
    # print(dd2)
    return dd2
#jieba
def Jieba(line='',num=0):
    segs=jieba.lcut(line)
    segs=filter(lambda x: len(x)>1,segs)
    segs=filter(lambda x:x not in stopwords,segs)
    list_tags=[]
    for tags in segs:
        if re.match('[0-9a-z\.A-Z%]+',tags):
            pass
        else:
            list_tags.append(tags)
    if num==0:
        return  list_tags
    elif num==1:
        list_str=' '.join(list_tags)
        return list_str
def originalSimHash(content):
    seg = Jieba(content,1)  # 分词
    keyWord = Tf_original.one_tfidf(seg, 20,1)  # 关键词提取
    keyList = []
    if keyWord==1:
        return 1
    else:
        for feature, weight in keyWord:  # 对关键词进行hash
            weight = Normalization(weight, keyWord[0][1], keyWord[1][1])  # 归一化倍数，参数设置
            feature = string_hash(feature)
            temp = [weight if i == '1' else -weight for i in feature]  # 将hash值用权值替代
            keyList.append(temp)
        list_sum = np.sum(np.array(keyList), axis=0)  # 20个权值列向相加
        if (keyList == []):  # 编码读不出来
            print ('00')
        simhash = ''
        # 权值转换成hash值
        for i in list_sum:
            if (i > 0):
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'
        return  simhash
def hammingDis(simhash1, simhash2):
    t1 = '0b' + simhash1#只要在数字前面加上0b的字符，就可以用二进制来表示十进制数了。
    t2 = '0b' + simhash2
    n = int(t1, 2) ^ int(t2, 2)#二进制到十进制转换 十进制按位异或
    i = 0
    while n:
        n &= (n - 1)
        i += 1
    if i<= 3:
       return  True
    else:
       return  False
def Normalization(x,Max,Min):
    if (Max/Min)>3 and x!=Max:
        return x*2.5
    else:
        return x
def string_hash(source):
    if source == "":
        return 0
    else:
        x = ord(source[0]) << 7#ord()它以一个字符（长度为1的字符串）作为参数，返回对应的ASCII数值
        m = 1000003
        mask = 2 ** 128 - 1
        for c in source:
            x = ((x * m) ^ ord(c)) & mask
        x ^= len(source)
        if x == -1:
            x = -2
        x = bin(x).replace('0b', '').zfill(64)[-64:]
        return str(x)
#   加载赤兔数据库里的simhash
def originalLoding():
    newid=16830994
    ms = MSSQL(host='192.168.128.7', user='app_yrbank_bd', pwd='Yhd$bCk#uC&qTpG8', db='yuanrongbank_bigdata')
    resList_num = ms.ExecQuery("select count(1) from ArticleCrawler where id >16830994 ",1)
    batch_size = 3000
    num_batch = int((resList_num[0][0] - 1) / batch_size) + 1
    for i in range(num_batch):
        resList = ms.ExecQuery("SELECT ID,content  FROM ArticleCrawler WHERE  ID>{0} and content IS not NULL  ORDER BY ID limit 3000".format(newid),1)
        for resList_line in resList:
            newid=resList_line[0]
            if resList_line is not None:
                simhash = originalSimHash(resList_line[1])
                sql2="update ArticleCrawler set simhash='{0}' where ID={1}".format(simhash,resList_line[0])
                try:
                    ms.ExecQuery(sql2,1)
                except pymysql.err.OperationalError:
                    time.sleep(10)
                    ms.ExecQuery(sql2, 1)
                logger.info('id:'+str(resList_line[0]))
    logger.info('历史SimHash总数={0}:加载至RecID={1}'.format(len(Sim_List), newid))
    return Sim_List
logger.info('Staring...')
logger.info('loding jieba and stopwords')
# jieba.load_userdict('')
stopwords = pd.read_csv(os.getcwd()+'/wordData/stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopword'],encoding='utf-8')
stopwords = stopwords['stopword'].values
Tf_original = tfidf.TfIdf()
Tf_original.loda_model(os.getcwd()+'/wordData/corpus_827W.txt')
logger.info('loding simhash  for origanal score')
Sim_List=originalLoding()
