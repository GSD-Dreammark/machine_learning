# 梯度下降法 （初遇）
# 设置步长为0.1 f_change为改变前后y值的变化 ，仅设置一个退出条件
import numpy as np
m=20
x0=np.ones((m,1))
# reshape 把数组重塑但不改变数据  reshape(m,1) m行 1列
x1=np.arange(1,m+1).reshape(m,1)
# hstack 函数  参数是元组  是增加维度的
x=np.hstack((x0,x1))
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
# 学习率
alpha=0.01
# 获取代价函数
def error_function(theta,X,y):
    diff=np.dot(X,theta)-y
    return (1./2*m)*np.dot(np.transpose(diff),diff)
# 获取梯度
def gradient_function(theta,X,y):
    # 矩阵相乘 np.dot
    # 矩阵乘法 行乘列
    diff=np.dot(X,theta)-y
    # 矩阵装置   对于一二维数组 直接.T即可装置 而高维数组采用transpose
    return (1./m)*np.dot(np.transpose(X),diff)
# 获取最小值
def gradient_descent(X,y,alpha):
    theta=np.array([1,1]).reshape(2,1)
    gradient=gradient_function(theta,X,y)
    print(gradient)
    # any 或运算 all 与运算   all(x1==x2) 快速判断两个array 是否相等  absolute 绝对值
    while not np.all(np.absolute(gradient)<=1e-5):
        theta =theta -alpha* gradient
        gradient =gradient_function(theta,X,y)
    return theta
optimal=gradient_descent(x,y,alpha)
print('optimal:',optimal)
print('error function:',error_function(optimal,x,y)[0,0])