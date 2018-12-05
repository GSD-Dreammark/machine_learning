import numpy as np
# 把字符串哈希化（hash)
def string_hash(source):
    if source == "":
        return 0
    else:
        # 把第一个字符取出获取ascll 码  左移7位(左移变大，右移变小)
        x = ord(source[0]) << 7#ord()它以一个字符（长度为1source的字符串）作为参数，返回对应的ASCII数值
        m = 1000003
        mask = 2 ** 128 - 1
        # ^ 按位异或运算符：当两对应的二进位相异时，结果为1
        # & 按位与运算符：参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
        for c in source:
            x = ((x * m) ^ ord(c)) & mask
        x^= len(source)
        if x == -1:
            x = -2
        # bin()返回一个整数 int 或者长整数 long int 的二进制表示
        # 0b是二进制
        # zfill  方法返回指定长度的字符串，原字符串右对齐，前面填充0。
        # 取出倒数64位
        x = bin(x).replace('0b', '').zfill(64)[-64:]
        return str(x)
# weight 词频 features 词组
def hash_number(weight,features):
    keyList=[]
    for feature in features:
        feature = string_hash(feature)
        # temp=[]
        # for i in feature:
        #     if i == '1':
        #         temp.append(weight)
        #     else:
        #         temp.append(-weight)
        #  == 注释的代码
        #   加权
        temp = [weight if i == '1' else -weight for i in feature]  # 将hash值用权值替代
        keyList.append(temp)
    #   合并
    list_sum = np.sum(np.array(keyList), axis=0)  # 20个权值列向相加
    if (keyList == []):  # 编码读不出来
        print ('00')
    simhash = ''
    # 降维
    # 权值转换成hash值
    for i in list_sum:
        if (i > 0):
            simhash = simhash + '1'
        else:
            simhash = simhash + '0'
    return  simhash