# 归一化数据  全不数据Stability
# 处理数据库中的数据
import pymysql
import time
from numpy import *
# class Mysql():
#     def __init__(self,host,user,pwd,db,charset):
#         self.host=host
#         self.user=user
#         self.pwd=pwd
#         self.db=db
#         self.charset=charset
#     def getConnection(self):
#         if not self.db:
#             raise(NameError,"没有设置数据库信息")
#         self.conn=pymysql.connect(host=self.host,user=self.user,password=self.pwd,database=self.db,charset=self.charset)
#         cur=self.conn.cursor()
#         if  not cur:
#             raise(NameError,"链接数据库失败")
#         else:
#             return cur
#     def ExecQuery(self,sql):
#         try:
#             cur=self.getConnection()
#             cur.execute(sql)
#             resList=cur.fetchall()
#         except pymysql.err.InternalError:
#             # if self.conn:
#             #     self.conn.close()
#             return
#         # finally:
#         #     if not self.conn:
#         #         self.conn.close()
#         return resList
#     def ExecNonQuery(self,sql):
#             cur = self.getConnection()
#             cur.execute(sql)
#             self.conn.commit()
#             self.conn.close()
class MSSQL:
    def __init__(self,host,user,pwd,db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db
    def __GetConnect(self):
        """
        得到连接信息
        返回: conn.cursor()
        """
        try:
            if not self.db:
                raise(NameError,"没有设置数据库信息")
            self.conn = pymysql.connect(host=self.host,user=self.user,password=self.pwd,database=self.db,charset='utf8')
            cur = self.conn.cursor()
            if not cur:
                raise(NameError,"连接数据库失败")
            else:
                return cur
        except pymysql.err.OperationalError:
            time.sleep(10)
            print('+++++++++++++++++++++++++++++')
            if not self.db:
                raise (NameError, "没有设置数据库信息")
            self.conn = pymysql.connect(host=self.host, user=self.user, password=self.pwd, database=self.db,
                                        charset='utf8')
            cur = self.conn.cursor()
            if not cur:
                raise (NameError, "连接数据库失败")
            else:
                return cur
    def ExecQuery(self,sql):
        """
        执行查询语句
        返回的是一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段
        调用示例：
                ms = MSSQL(host="localhost",user="sa",pwd="123456",db="PythonWeiboStatistics")
                resList = ms.ExecQuery("SELECT id,NickName FROM WeiBoUser")
                for (id,NickName) in resList:
                    print str(id),NickName
        """
        cur = self.__GetConnect()
        cur.execute(sql)
        resList = cur.fetchall()
        #查询完毕后必须关闭连接
        cur.close()
        self.conn.close()
        return resList
    def ExecNonQuery(self,sql):
        """
        执行非查询语句
        调用示例：
            cur = self.__GetConnect()
            cur.execute(sql)
            self.conn.commit()
            self.conn.close()
        """
        cur = self.__GetConnect()
        cur.execute(sql)
        self.conn.commit()
        cur.close()
        self.conn.close()
        return  cur.lastrowid
# 矩阵归一化数据
def autoNorm_use(dataSet):
    # 获取第一列的最小值
    minVals = dataSet[:,0].min(0)
    # 获取第一列的最大值
    maxVals=dataSet[:,0].max(0)
    ranges=maxVals-minVals
    # 构造一个dataSet 行，列的全0矩阵
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet[:,0] -tile(minVals, (1, m))[0]
    # 特征值相除
    normDataSet=(normDataSet/tile(ranges,(1,m))[0])*100
    return normDataSet ,ranges,minVals
def normalize(max_number,min_number,number):
    diff = float(max_number - min_number)
    # for data in datas:
    if diff != 0:
        Stability = (float(number - min_number) / diff)*100.0
    else:
        Stability = 0
    return Stability
def Normalization(sql1,sql2,my):
    # sql = "SELECT Stability,WeixinId,ContentQuality,OriginalProportion,Transmissible,TotalIndex FROM `XinBangCount`"
    # my=Mysql(host='192.168.3.20', user='dev_tangzheng', pwd='v7mk%fKl6T', db='yuanrongbank',charset='utf8')
    datas=my.ExecQuery(sql1)
    Nor_list=[]
    for data in datas:
        if data[0]==None:
            continue
        if data[0]=='Null':
            continue
        Nor_list.append(data[0])
    i=0
    Score=[]
    for data in datas:
        #  查出的数据有None值
        if data[0]==None:
            continue
        if data[0]=='Null':
            continue
        Score=normalize(max(Nor_list),min(Nor_list),data[0])
        # if i==0:
        #     Score=autoNorm_use(array(datas))
        # sq2=sql2.format(Score[0][i],data[1])
        sq2 = sql2.format(Score, data[1])
        print(sq2)
        my.ExecNonQuery(sq2)
        i+=1
def specialization_and_coreValuesScore_main(my):
    time1 = time.time()
    sql1='SELECT name from DictInfo WHERE Type=13'

    # my = MSSQL(host='192.168.3.20', user='dev_tangzheng', pwd='v7mk%fKl6T', db='yuanrongbank')

    # my = MSSQL(host='192.168.128.7', user='app_yuanrongbank', pwd='kP#^C03EwljPL7s', db='yuanrongbank')
    datas = my.ExecQuery(sql1)
    for data in datas:
        # print(data[0])
        sql= "SELECT specializationScore,Id FROM `XinBang_5000` where Category='{0}'".format(data[0])
        sql2 = "update `XinBang_5000` set specializationScore={0} where Id='{1}'"
        Normalization(sql,sql2,my)
    sql3="SELECT coreValuesScore,Id FROM `XinBang_5000` where statusEvaluation>=1"
    sql4="update `XinBang_5000` set coreValuesScore={0} where Id='{1}'"
    Normalization(sql3, sql4, my)
    time2 = time.time()
    print('时间', time2 - time1)
def themeScore_main(my):
    time1 = time.time()
    # my = MSSQL(host='192.168.3.20', user='dev_tangzheng', pwd='v7mk%fKl6T', db='yuanrongbank')
    # my = MSSQL(host='192.168.128.7', user='app_yuanrongbank', pwd='kP#^C03EwljPL7s', db='yuanrongbank')
    sql3 = "SELECT themeScore,Id FROM `XinBang_5000` where statusEvaluation>=1"
    sql4 = "update `XinBang_5000` set themeScore={0} where Id='{1}'"
    Normalization(sql3, sql4, my)
    time2 = time.time()
    print('时间', time2 - time1)
def contentEvaluationScore_main(my):
    time1 = time.time()
    # my = MSSQL(host='192.168.3.20', user='dev_tangzheng', pwd='v7mk%fKl6T', db='yuanrongbank')
    # my = MSSQL(host='192.168.128.7', user='app_yuanrongbank', pwd='kP#^C03EwljPL7s', db='yuanrongbank')
    sql3 = "SELECT contentEvaluationScore,Id FROM `XinBang_5000` where statusEvaluation>=1"
    sql4 = "update `XinBang_5000` set contentEvaluationScore={0} where Id='{1}'"
    Normalization(sql3, sql4, my)
    time2 = time.time()
    print('时间', time2 - time1)
if __name__=="__main__":
    # themeScore_main()
    # contentEvaluationScore_main()
    my = MSSQL(host='192.168.3.20', user='dev_tangzheng', pwd='v7mk%fKl6T', db='yuanrongbank')
    # my = MSSQL(host='192.168.128.7', user='app_yuanrongbank', pwd='kP#^C03EwljPL7s', db='yuanrongbank')
    specialization_and_coreValuesScore_main(my)
    # themeScore_main(my)
    # contentEvaluationScore_main(my)



