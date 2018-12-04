from functools import reduce
import random
import numpy
import math
import struct
from datetime import datetime
# 反向传播算法(不针对向量)
# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
def sigmoid(inX):
    return 1.0/(1+numpy.exp(-inX))
class Node(object):
    def __init__(self,layer_index,node_index):
        '''
               构造节点对象。
               layer_index: 节点所属的层的编号
               node_index: 节点的编号
               '''
        self.layer_index=layer_index
        self.node_index=node_index
        # 存的是conn
        self.downstream=[]
        self.upstream=[]
        self.output=0
        self.delta=0

    # 设置节点的输出值。如果节点属于输入层会用到这个函数。
    def set_output(self,output):
        self.output=output

    # 添加一个到下游节点的连接
    def append_downstream_connection(self,conn):
        self.downstream.append(conn)

    # 添加一个到上游节点的连接
    def append_upstream_connection(self,conn):
        self.upstream.append(conn)
    # 根据y=sigmoid(w^T*x) w是权重，w^T是权重矩阵 x是向量 计算节点的输出
    def calc_output(self):
        # 最后一个参数是ret的初始值
        output=reduce(lambda ret,conn:ret+conn.upstream_node.output*conn.weight,self.upstream,0)
        self.output=sigmoid(output)

    # 节点属于隐藏层时，根据式4计算delta
    def calc_hidden_layer_delta(self):
        downstream_delta=reduce(lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,self.downstream,0.0)
        self.delta=self.output*(1-self.output)*downstream_delta

    # 节点属于输出层时，根据式3计算delta
    def calc_output_layer_delta(self,label):
        self.delta=self.output*(1-self.output)*(label-self.output)

    # 打印节点的信息
    def __str__(self):
        node_str='{0}-{1}:output{2} delta:{3}'.format(self.layer_index,self.node_index,self.output,self.delta)
        downstream_str=reduce(lambda ret,conn:ret+'\n\t' + str(conn), self.downstream, '')
        upstream_str=reduce(lambda ret,conn:ret+'\n\t' + str(conn), self.upstream, '')
        return node_str+'\n\tdownstream:'+downstream_str+'\n\tupstream:'+upstream_str
#     为了实现一个输出恒为1的节点(计算偏置项时需要)
class ConstNode(object):
    def __init__(self,layer_index,node_index):
        '''
               构造节点对象。
               layer_index: 节点所属的层的编号
               node_index: 节点的编号
               '''
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.output=1

    # 添加一个到下游节点的连接
    def append_downstream_connection(self,conn):
        self.downstream.append(conn)

    # 节点属于隐藏层时，根据式4计算delta
    def calc_hidden_layer_delta(self):
        downstream_delta=reduce(lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,self.downstream,0.0)
        self.delta=self.output*(1-self.output)*downstream_delta

    # 打印节点的信息
    def __str__(self):
        node_str='{0}-{1}:output{2}'.format(self.layer_index,self.layer_index,self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str
# Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作。
class Layer(object):
    def __init__(self,layer_index,node_count):
        '''
               初始化一层
               layer_index: 层编号
               node_count: 层所包含的节点个数
               '''
        self.layer_index=layer_index
        self.nodes=[]
        for i in range(node_count):
            self.nodes.append(Node(layer_index,i))
        # 偏置项节点（恒唯一）
        self.nodes.append(ConstNode(layer_index,node_count))

    # 设置层的输出。当层是输入层时会用到。
    def set_output(self,data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    # 计算层的输出向量
    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    # 打印层的信息
    def dump(self):
        for node in self.nodes:
            print(node)
class Connection(object):
    def __init__(self,upstream_node,downstream_node):
        '''
               初始化连接，权重初始化为是一个很小的随机数
               upstream_node: 连接的上游节点
               downstream_node: 连接的下游节点
               '''
        self.upstream_node=upstream_node
        self.downstream_node=downstream_node
        self.weight=random.uniform(-0.1,0.1)
        self.gradient=0.0

    # 计算梯度
    def calc_gradient(self):
        self.gradient=self.downstream_node.delta*self.upstream_node.output

    # 获取当前梯度
    def get_gradient(self):
        return self.gradient

    # 根据梯度下降算法更新权重
    def update_weight(self,rate):
        self.calc_gradient()
        self.weight+=rate*self.gradient
    def __str__(self):
        '''
               打印连接信息
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)
class Connections(object):
    def __init__(self):
        self.connections=[]
    def add_connection(self,connection):
        self.connections.append(connection)
    def dump(self):
        for conn in self.connections:
            print(conn)
class Network(object):
    def __init__(self,layers):
        '''
                初始化一个全连接神经网络
                layers: 二维数组，描述神经网络每层节点数
                '''
        self.connections=Connections()
        self.layers=[]
        layer_count=len(layers)
        # node_count=0
        # 添加层数和每层的节点数
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i]))
        #   链接数（节点和节点间的链接）
        for layer in range(layer_count-1):
            connections=[Connection(upstream_node,downstream_node) for upstream_node in self.layers[layer].nodes
                         # 去除这行最后一个
                         for downstream_node in self.layers[layer+1].nodes[:-1]]
            print()
            for conn in connections:
                self.connections.add_connection(conn)
                # 添加conn的下一层的上一层节点
                conn.downstream_node.append_upstream_connection(conn)
                # 添加conn的这一层的下一层节点
                conn.upstream_node.append_downstream_connection(conn)
    def train(self,labels,data_set,rate,iteration):
        '''
                训练神经网络
                labels: 数组，训练样本标签。每个元素是一个样本的标签。
                data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
                iterator 迭代次数
                '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)

    # 内部函数，用一个样本训练网络
    def train_one_sample(self,label,sample,rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    # 内部函数，计算每个节点的delta
    def calc_delta(self,label):
        output_nodes=self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        # [-2::-1] -2 步长为2 ，-1 倒序开始
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    # 内部函数，更新每个连接权重
    def update_weight(self,rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    # 内部函数，计算每个连接的梯度
    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self,label,sample):
        '''
              获得网络在一个样本下，每个连接上的梯度
              label: 样本标签
              sample: 样本输入
              '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self,sample):
        '''
               根据输入的样本预测输出值
               sample: 数组，样本的特征，也就是网络的输入向量
               '''
        self.layers[0].set_output(sample)
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node:node.output,self.layers[-1].nodes[:-1]))

    # 打印网络信息
    def dump(self):
        for layer in self.layers:
            layer.dump()
class Normalizer(object):
    def __init__(self):
        # self.mask=[ 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
        self.mask = [0x1, 0x2]
    def norm(self,number):
        return list(map(lambda m:0.9 if number&m else 0.1,self.mask))
    def denorm(self,vec):
        binary=list(map(lambda i:1 if i>0.5 else 0,vec))
        for i in range(len(self.mask)):
            binary[i]=binary[i]*self.mask[i]
        return reduce(lambda x,y:x+y,binary)
def gradient_check(network,sample_feature,sample_label):
    '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''
    # 计算网络误差
    network_error=lambda vec1,vec2:0.5*reduce(lambda a,b:a+b,map(lambda v:(v[0]-v[1])*(v[0]-v[1]),zip(vec1,vec2)))
    # 获取网络在当前样本下的每个链接的梯度
    network.get_gradient(sample_label,sample_feature)
    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取制定链接的梯度
        actual_gradient=conn.get_gradient()
        # 增加一个很小的值，计算网络的误差
        epsilon=0.0001
        conn.weight+=epsilon
        error1=network_error(network.predict(sample_feature),sample_label)
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
        # 打印
        print('expected gradient: \t%f\n actual gradient: \t%f' % (expected_gradient, actual_gradient))

def train_data_set():
    normalizer=Normalizer()
    data_set=[]
    labels=[]
    for i in range(0,256,8):
        n=normalizer.norm(int(random.uniform(0,256)))
        data_set.append(n)
        labels.append(n)
    return labels,data_set
def train(network):
    labels,data_set=train_data_set()
    network.train(labels,data_set,0.3,50)
def test(network,data):
    normalizer=Normalizer()
    norm_data=normalizer.norm(data)
    predict_data=network.predict(norm_data)
    print('\ttestdata(%u)\tpredict(%u)' % (data, normalizer.denorm(predict_data)))
def correct_ratio(network):
    normalizer=Normalizer()
    correct=0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i)))==i:
            correct+=1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))

def gradient_check_test():
    net = Network([2, 2, 2])
    train(net)
    # net.dump()
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    # net = Network([8, 3, 8])
    # train(net)
    # net.dump()
    # correct_ratio(net)
    gradient_check_test()



#
# class Loader(object):
#     def __init__(self,path,count):
#         '''
#                 初始化加载器
#                 path: 数据文件路径
#                 count: 文件中的样本个数
#                 '''
#         self.path=path
#         self.count=count
#     def get_file_content(self):
#         '''读取文件内容'''
#         f=open(self.path,'rb')
#         content=f.read()
#         f.close()
#         return content
#     def to_int(self,byte):
#         '''内部函数，从文件中获取图像'''
#         return struct.unpack('B',byte)[0]
#
# class ImageLoader(Loader):
#     #  内部函数，从文件中获取图像
#     def get_picture(self,content,index):
#         start=index*28*28+16
#         picture=[]
#         for i in range(28):
#             picture.append([])
#             for j in range(28):
#                 picture[i].append(self.to_int(content[start+i*28+j]))
#         return picture
#
#     # 内部函数，将图像转化为样本的输入向量
#     def get_one_sample(self,picture):
#         sample=[]
#         for i in range(28):
#             for j in range(28):
#                 sample.append(picture[i][j])
#         return sample
#     # 加载数据文件，获得全部样本的输入向量
#     def load(self):
#         content=self.get_file_content()
#         data_set=[]
#         for index in range(self.count):
#             data_set.append(self.get_one_sample(self.get_picture(content,index)))
#         return data_set
# # 标签数据加载器
# class LabelLoader(Loader):
#     # 加载数据文件，获得全部样本的标签向量
#     def load(self):
#         content=self.get_file_content()
#         labels=[]
#         for index in range(self.count):
#             labels.append(self.norm(content[index+8]))
#         return labels
#     def norm(self,label):
#         label_vec=[]
#         label_value=self.to_int(label)
#         for i in range(10):
#             if i==label_value:
#                 label_vec.append(0.9)
#             else:
#                 label_vec.append(0.1)
#         return label_vec
# def get_training_data_set():
#     '''获得训练数据集'''
#     image_loader = ImageLoader('train-images-idx3-ubyte', 60000)
#     label_loader = LabelLoader('train-labels-idx1-ubyte', 60000)
#     return image_loader.load(), label_loader.load()
# def get_test_data_set():
#     '''
#     获得测试数据集
#     '''
#     image_loader = ImageLoader('t10k-images-idx3-ubyte', 10000)
#     label_loader = LabelLoader('t10k-labels-idx1-ubyte', 10000)
#     return image_loader.load(), label_loader.load()
# # 网络的输出是一个10维向量，这个向量第个(从0开始编号)元素的值最大，那么就是网络的识别结果
# def get_result(vec):
#     max_value_index=0
#     max_value=0
#     for i in range(len(vec)):
#         if vec[i]>max_value:
#             max_value=vec[i]
#             max_value_index=i
#     return max_value_index
# # 使用错误率来对网络进行评估
# def evaluate(network,test_data_set,test_labels):
#     error=0
#     total=len(test_data_set)
#     for i in range(total):
#         label=get_result(test_labels[i])
#         predict=get_result(network.predict(test_data_set[i]))
#         if label !=predict:
#             error +=1
#     return float(error)/float(total)
# def train_and_evaluate():
#     last_error_ratio=1.0
#     epoch=0
#     train_data_set,train_labels=get_training_data_set()
#     test_data_set,test_labels=get_test_data_set()
#     network=Network([789,300,10])
#     while True:
#         epoch+=1
#         network.train(train_labels,train_data_set,0.3,1)
#         # print("&s epoch %d finished"%(now(),epoch))
#         if epoch%10==0:
#             error_ratio=evaluate(network,test_data_set,test_labels)
#             # print('%s after epoch %d, error ratio is %f' % (now(), epoch, error_ratio))
#             if error_ratio>last_error_ratio:
#                 break
#             else:
#                 last_error_ratio=error_ratio
#
# if __name__=='__main__':
#     train_and_evaluate()
