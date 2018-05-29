# -*- coding: utf-8 -*-
#import pandas as pd
#def print_full(x):
#    pd.set_option('display.max_rows',  len(x))
#    print(x)
#    pd.reset_option('display.max_rows')

from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris() #python自带iris数据集

#print iris

knn.fit(iris.data,iris.target)
predictedLabel = knn.predict([[0.1,0.2,0.3,0.4]])
 
print predictedLabel
