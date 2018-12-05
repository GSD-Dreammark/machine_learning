# https://blog.csdn.net/qq_16912257/article/details/72156277  实体python操作
# https://blog.csdn.net/lihaitao000/article/details/52355704  理解
from simhash import Simhash,SimhashIndex
# 查看simhash值
print(Simhash('I am very happy'.split()).value)
# 计算Simhash的距离
hash1=Simhash('I am very happy'.split())
hash2=Simhash('I am very sad'.split())
distance=hash1.distance(hash2)
print(distance)
# 对于文本数据特别大时使用
# 建立索引
data = {
    '1': 'How are you I Am fine . blar blar blar blar blar Thanks .'.lower().split(),
    '2': 'How are you i am fine .'.lower().split(),
    '3': 'This is simhash test .'.lower().split(),
}
objs=[(id,Simhash(sent)) for id,sent in data.items()]
print(objs)
# k是容忍度 ，k越大，检索出的相似文本就越多
index=SimhashIndex(objs,k=20)
# 检索
s1=Simhash('How are you ,blar blar blar blar Thanks bhd'.lower().split())
print(index.get_near_dups(s1))
# 增加新索引
index.add('4',s1)