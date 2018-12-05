import numpy
# mat()函数中数据可以为字符串以分号(；)分割，或者为列表形式以逗号（，）分割
# array 只支持列表形式以逗号（，）分割
m1=numpy.mat('1,2;3,4')
print(m1)
m2=numpy.mat([[1,3],[2,4]])
print(m2)
a1=numpy.array([[1,3],[2,4]])
print(a1)
# 矩阵乘机 mat() 有* ，。dot() 矩阵对应位置元素相乘需调用numpy.multiply()函数。
# array()函数中矩阵的乘积只能使用 .dot()函数。而星号乘 （*）则表示矩阵对应位置元素相乘，与numpy.multiply()函数结果相同
print(m1*m2)
print(m1.dot(m2))
print(a1.dot(m2))
print(m2)
# 按行相加 sum加矩阵时
print(sum(m2))
# numpy.sum （或者Numpy中的sum函数），无参时，所有全加；axis=0，按列相加；axis=1，按行相加
print(numpy.sum(m2))