import numpy as np
# array 与 matrix
c=np.mat([[0,6,5],[0,0,1]]) #将列表矩阵化
c1=np.array([[1,2,3],[4,5,6]]) #将列表数组化，数组的每个元素是原列表的元素
print(type(c.A),type(c1))
print(c[0])
