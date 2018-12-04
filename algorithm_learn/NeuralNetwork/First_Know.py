from functools import reduce
import numpy
# 初探神经网路
class Perception(object):
    def __init__(self,input_num,activator):
        '''
               初始化感知器，设置输入参数的个数，以及激活函数。
               激活函数的类型为double -> double
               激活函数是activator
        '''
        self.activator=activator
        # 权重向量初始化
        self.weights=[0.0 for _ in range(input_num)] # 生成input_num 个0.0
        # 偏置项初始化
        self.bias=0.0
    def __str__(self):
        # 打印学到的权重，偏置项
        return 'weights :{},bias:{}'.format(self.weights,self.bias)
    def predict(self,input_vec):
        # 输入向量，输出感知器的计算结果
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        # zip函数接受任意多个可迭代对象作为参数, 将对象中对应的元素打包成一个tuple
        # map()是Python内置的高阶函数，它接收一个函数f和一个list，并通过把函数f依次作用在list的每个元素上，得到一个新的list并返回
        # reduce()函数会对参数序列中元素进行累积(通过某个函数进行累积如add函数)reduce(function, iterable[, initializer])function -- 函数，有两个参数iterable -- 可迭代对象
        # return self.activator(reduce(lambda a, b: a + b,map(lambda x_w: x_w[0] * x_w[1],zip(input_vec, self.weights)), 0.0) + self.bias)
        # Caresult = reduce(lambda a, b: a + b,map(lambda x_w: x_w[0] * x_w[1],zip(input_vec, self.weights)), 0.0) + self.bias
        Caresult = numpy.sum(numpy.array(input_vec) * numpy.array(self.weights), 0) + self.bias
        return self.activator(Caresult)
    def train(self,input_vecs,labels,iteration,rate):
        """ 输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率"""
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)
    def _one_iteration(self,input_vecs,labels,rate):
        '''一次迭代，把所有的训练数据过一遍'''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples=zip(input_vecs,labels)
        for (input_vec,label) in samples:
            # 计算感知器在当前权重下的输出
            output=self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec,output,label,rate)
    # 循环迭代权重和阈值
    def _update_weights(self,input_vec,output,label,rate):
        '''按照感知器规则更新权重 '''
        delta=label-output
        # python2 lambda (x,y) python3 lambda x_w:x_w[0]*x_w[1]
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        self.weights=list(map(lambda w_x:w_x[1]+rate*delta*w_x[0],zip(input_vec,self.weights)))
        # 更新bias
        self.bias+=rate*delta
#  定义激活函数f
def f(x):
    return 1 if x>0 else 0
#  基于and真值表构建训练数据
def get_training_dataset():
    # 构建训练数据
    # 输入向量列表
    input_vecs=[[1,1], [0,0], [1,0], [0,1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels=[1,0,0,0]
    return input_vecs,labels
# 使用and真值表训练感知器
def train_and_perceptron():
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p=Perception(2,f)
    input_vec,labels,=get_training_dataset()
    # 训练，迭代10轮, 学习速率为0.1
    p.train(input_vec,labels,10,0.1)
    # 返回训练好的感知器
    return p

if __name__=='__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))
