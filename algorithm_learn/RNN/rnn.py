import numpy as np
from functools import reduce
# RecurrentLayer类来实现一个循环层（只有循环层）
# 激活函数
class ReluActivator(object):
    # 激活函数本身
    def forward(self,weighted_input):
        return max(0,weighted_input)
    # 激活函数的导数
    def backward(self,output):
        return 1 if output>0 else 0
# 激活函数
class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input
    # 函数的导数
    def backward(self, output):
        return 1
# 对numpy数组进行element wise操作   修改矩阵的值 把矩阵的值修改为调用玩激活函数后的值
def element_wise_op(array,op):
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...]=op(i)
class RecurrentLayer(object):
    def __init__(self,input_width,state_width,activator,learning_rate):
        self.input_width=input_width
        self.state_width=state_width
        self.activator=activator
        self.learning_rate=learning_rate
        self.times=0 # 当前时刻初始化为t0
        self.state_list=[] # 保存各个时刻的state
        self.state_list.append(np.zeros((state_width,1))) # 初始化s0
        self.U=np.random.uniform(-1e-4,1e-4,(state_width,input_width)) # 初始化U
        self.W=np.random.uniform(-1e-4,1e-4,(state_width,state_width)) # 初始化W
    # 计算每个时刻s的取值
    def forward(self,input_array):
        self.times+=1
        # U*x+W*St-1
        state=(np.dot(self.U,input_array)+np.dot(self.W,self.state_list[-1]))
        element_wise_op(state,self.activator.forward)
        self.state_list.append(state)

    def backward(self,sensitivity_array,activator):
        '''实现BPTT算法'''
        self.calc_delta(sensitivity_array,activator)
        self.calc_gradient()

    def calc_delta(self,sensitivity_array,activator):
        self.delta_list=[] #用来保存各个时刻的误差项
        for i in range(self.times):
            # 初始化各个时刻的误差项为0
            self.delta_list.append(np.zeros((self.state_width,1)))
        self.delta_list.append(sensitivity_array)
        # 迭代每个时刻的误差项 倒序
        for k in range(self.times-1,0,-1):
            self.calc_delta_k(k,activator)
    def calc_delta_k(self,k,activator):
        # 公式三
        # 根据k+1时刻的delta计算k时刻的delta
        # state = self.state_list[k + 1].copy()
        state=self.state_list[k].copy()
        element_wise_op(state, activator.backward)
        # element_wise_op(self.state_list[k+1],activator.backward)
        self.delta_list[k]=np.dot(np.dot(self.delta_list[k+1].T,self.W),np.diag(state[:,0])).T
    def calc_gradient(self):
        self.gradient_list=[] #保存各个时刻的权重梯度
        for t in range(self.times+1):
            self.gradient_list.append(np.zeros((self.state_width,self.state_width)))
        for t in range(self.times,0,-1):
            self.calc_gradient_t(t)
        # 实际的梯度是各个时刻梯度之和
        self.gradient=reduce(lambda a,b:a+b,self.gradient_list)
    def calc_gradient_t(self,t):
        '''计算每个时刻t权重的梯度'''
        # 式5
        # 此处转置是为了方便计算梯度
        gradient=np.dot(self.delta_list[t],self.state_list[t-1].T)
        self.gradient_list[t]=gradient

    def update(self):
        # 按照梯度下降，更新权重
        self.W-=self.learning_rate*self.gradient

    def reset_state(self):
        self.times=0 #当前时刻初始化为t0
        self.state_list=[] # 保存各个时刻的state
        self.state_list.append(np.zeros((self.state_width,1))) # 初始化s0

def data_set():
    # 一个列表有两个矩阵
    x=[np.array([[2],[5],[3],[7]]),np.array([[6],[3],[4],[8]]),np.array([[7],[8],[4],[9]])]
    # 把数组转化为矩阵]
    d=np.array([[1],[2]])
    return x,d
#     梯度检查
def gradient_check():
    error_function=lambda o:o.sum()
    r1=RecurrentLayer(4,3,IdentityActivator(),1e-3)
    # 计算forward值
    x,d=data_set()
    r1.forward(x[0])
    r1.forward(x[1])
    r1.forward(x[2])
    # 求取sensitivity map
    sensitity_array=np.ones(r1.state_list[-1].shape,dtype=np.float64)
    # 计算梯度
    r1.backward(sensitity_array,IdentityActivator())
    # 检查梯度
    epsilon=10e-4
    for i in range(r1.W.shape[0]):
        for j in range(r1.W.shape[1]):
            r1.W[i,j]+=epsilon
            r1.reset_state()
            r1.forward(x[0])
            r1.forward(x[1])
            r1.forward(x[2])
            err1=error_function(r1.state_list[-1])
            r1.W[i,j]-=2*epsilon
            r1.reset_state()
            r1.forward(x[0])
            r1.forward(x[1])
            r1.forward(x[2])
            err2=error_function(r1.state_list[-1])
            expect_grad=(err1-err2)/(2*epsilon)
            r1.W[i,j]+=epsilon
            print('weights(%d,%d): expected - actural %f - %f' % (i, j, expect_grad, r1.gradient[i,j]))

def test():
    l=RecurrentLayer(3,2,ReluActivator(),1e-3)
    x,d=data_set()
    l.forward(x[0])
    l.forward(x[1])
    l.backward(d,ReluActivator())
    return l

if __name__ == '__main__':
    gradient_check()
