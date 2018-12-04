# 词频向量化

from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer 类会将文本中的词语转换为词频矩阵，例如矩阵中包含一个元素a[i][j]，它表示j词在i类文本下的词频。
# 它通过 fit_transform 函数计算各个词语出现的次数，通过get_feature_names()可获取词袋中所有文本的关键字，
# 通过 toarray()可看到词频矩阵的结果
vectorizer = CountVectorizer(min_df=1)

corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?',
          ]
X = vectorizer.fit_transform(corpus)
feature_name = vectorizer.get_feature_names()
print (X)
print (feature_name)
print (X.toarray())
