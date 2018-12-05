import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec
sentences=[['first','sentence'],['second','sentence','is']]
# 模型训练
model =gensim.models.Word2Vec(sentences,min_count=1)
# min_count,频数阈值，大于等于1的保留
# size，神经网络 NN 层单元数，它也对应了训练算法的自由程度
# workers=4，default = 1 worker = no parallelization 只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。
# 第二种训练方式
new_model = gensim.models.Word2Vec(min_count=1)  # 先启动一个空模型 an empty model
new_model.build_vocab(sentences)                 # can be a non-repeatable, 1-pass generator
new_model.train(sentences, total_examples=new_model.corpus_count, epochs=new_model.iter)
# can be a non-repeatable, 1-pass generator
# sentences=word2vec.Text8Corpus("分词后的爽肤水评论.txt")
# model=word2vec.Word2Vec(sentences,size=50)
# y2=model.similarity("好","还行")
# print(y2)
# for i in model.most_similar("滋润"):
#     print(i[0],i[1])
# 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5,
# 第三个参数是神经网络的隐藏层单元数，默认为100
# 训练模型
# model=word2vec.Word2Vec(sentences,min_count=5,size=50)
# 模型使用
# 根据词向量求相似性
model.similarity('first','id') # 两个词的相似性距离
# 计算相似的词，topn＝1设置自取最相似词表中的第一个
model.most_similar(positive=['first','second'],negative=['sentence'],topn=1)
# 找出不匹配的词语
model.doesnt_match("input is lunch he sentence cat".split())
# 词向量查询
model['first']
# 模型的导出和导入
model.save('/tmp/mymodel')
new_model=gensim.models.Word2Vec.load('/tmp/mymodel')
# 载入 .txt文件
odel=Word2Vec.load_word2vec_format('/tmp/vectors.txt',binary=False)
# 载入 .bin文件
model=Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz',binary=True)
# 训练模型
word2vec=gensim.models.word2vec.Word2Vec(sentences(),size=256,window=10,min_count=64,sg=1,hs=1,iter=10,workers=25)
# 导出模型
word2vec.save('wordvec_wx')
# 其他导入方法
# import numpy
# word_2x = numpy.load('xxx/word2vec_wx.wv.syn0.npy')
# from gensim.models.keyedvectors import KeyedVectors
# word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
# word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format
# 增量训练
# model=gensim.models.Word2Vec.load('/tmp/mymodel')
# model.train(more_sentences)
# ************************************** eg:
# model=gensim.models.Word2Vec.load(temp_path)
# more_sentences=[['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']]
# model.build_vocab(more_sentences, update=True)
# model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)