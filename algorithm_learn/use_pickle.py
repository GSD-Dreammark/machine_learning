# try:
#     import cPickele as pickle
# except:
#     import pickle
# import pprint
# # pprint 提供了打印出任何python数据结构类和方法
# info=[1,2,3,'abc','iplaypython']
# print('原始数据')
# pprint.pprint(info)
# data1=pickle.dumps(info)
# data2=pickle.loads(data1)
# print("序列化：%r" %data1)
# print("反序列化：%r" %data2)
# 在Pickle模块中有2个常用的函数方法，一个叫做dump()，另一个叫做load()。
# 第三部分， pickle.dump()方法：
# 这个方法的语法是：pickle.dump(对象, 文件，[使用协议])
# 提示：将要持久化的数据“对象”，保存到“文件”中，使用有3种，索引0为ASCII，1是旧式2进制，2是新式2进制协议，不同之处在于后者更高效一些。
# 默认的话dump方法使用0做协议。
# 使用pickle模块将数据对象保存到文件
# import pickle
# data1={'a':[1,2.0,3,4+6j],
#        'b':('string',u'Unicode string'),
#        'c':None}
# selfref_list=[1,2,3]
# selfref_list.append(selfref_list)
# output=open('data.pkl','wb')
# #  Pickle dictionary using proftocol 0
# pickle.dump(data1,output)
# # Pickle the list using the highest protocol available
# pickle.dump(selfref_list,output,-1)
# output.close()
# 使用pickle模块从文件中重构python对象
import pprint,pickle
pkl_file=open('data.pkl','rb')
data1=pickle.load(pkl_file)
pprint.pprint(data1)
data2=pickle.load(pkl_file)
pprint.pprint(data2)
pkl_file.close()