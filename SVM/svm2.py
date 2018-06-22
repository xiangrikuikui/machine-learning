# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
from sklearn import svm

#we create 40 separable points
np.random.seed(0) #每次运行程序时抓取的随机值相同
#X为训练实例，通过正太分布的方式产生20行2列的数
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
#Y 为分类标记，前20个为1，后20个为1
Y = [0] * 20 + [1] * 20

#fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X,Y)

#get the separating hyperplane
w = clf.coef_[0] #w是2维的
a = -w[0] / w[1] #a代表画出的点的斜率
xx = np.linspace(-5,5) #从-5到5产生连续的x值
#(clf.intercept_[0]) / w[1]为所画直线的截距
yy = a * xx - (clf.intercept_[0]) / w[1] 

#plot the parallels to the separating hyperplane that pass through the
#support vectors
b = clf.support_vectors_[0] #取一个支持向量
yy_down = a * xx + (b[1] - a * b[0]) #(b[1] - a * b[0])为截距
b = clf.support_vectors_[-1] #取最后一个支持向量
yy_up = a * xx + (b[1] - a * b[0])

print('w: ',w)
print('a: ',a)
print('xx: ',xx)
print('yy: ',yy)
print('support vectors: ',clf.support_vectors_)
print('clf.coef_: ',clf.coef_)


pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')

#pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
#           s=80,facecolors='none')
pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired) #画出两类点


pl.axis('tight')  #使坐标系的最大值和最小值和我设定的坐标范围一致
pl.show()