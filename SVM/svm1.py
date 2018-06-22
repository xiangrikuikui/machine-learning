# -*- coding: utf-8 -*-
from sklearn import svm
X = [[2,0],[1,1],[2,3]]
y = [0,0,1] #将上面的点分类标记
clf = svm.SVC(kernel = 'linear') #调用svm中的SVC方程 ，选择线性核函数
clf.fit(X,y)  #fit中X代表实例， y为X对应的分类标记

#此时已算出分类器
print("clf: ", clf)

#get support vectors
print("support vectors: ", clf.support_vectors_)

#get indices of support vectors
print("indices of support vectors: ", clf.support_)

#get number of support vectors for each class
print("number of support vectors for each class: ", clf.n_support_)

#predict another point class
print("predict [2,0]: ", clf.predict([[2,0]]))