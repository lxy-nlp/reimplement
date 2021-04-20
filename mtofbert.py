# !/home/lxy/anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/13 下午8:54
# @Author  : lxy
# @FileName: mtofbert.py
# @Software: PyCharm


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.linalg import norm

def tf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

s1 = '你在干嘛呢'
s2 = '你在干什么呢'
print(tf_similarity(s1, s2))



#  标题 同一个城市 用tfidf
if __name__ == '__main__':
    # 按照 城市分类 以厦门为重点对象 因为厦门的数量最多 最为可信
    # corpus = [] 所有的文本加起来
    # 计算每2个文本之间的相似度
    # 标题之间的相似度