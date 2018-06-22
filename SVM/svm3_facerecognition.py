# -*- coding: utf-8 -*-
from __future__ import print_function
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

print(__doc__)

#Display progress logs on stdout
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')



#Download the data, if not already on disk and load it as numpy arrays
#fetch_lfw_people(): load The Labeled Faces in the Wild face recognition dataset

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

#introspect the images arrays to find the shapes (for plotting)

n_samples, h, w = lfw_people.images.shape  #images.shape返回实例个数n_samples,h,w
print("n_samples: %d, h: %d, w: %d" % (n_samples, h, w))
#for machine learning we use 2 data directly (as relative pixel)
#positions info is ignored by this model

X = lfw_people.data  #X每一行是一个实例，每一列是一个特征
n_features = X.shape[1]  #1代表列数，返回矩阵的列数即特征向量的维度
#n_feature=h*w=50*37=1850

#the label to predict is the id of the person
y = lfw_people.target  #调用target属性，返回每一个实例的分类标记
target_names = lfw_people.target_names  #返回所有类别的名字
n_classes = target_names.shape[0]  #返回类别的人数，即有多少个人要进行识别


print("Total dataset size: ")
print("n_samples: %d" % n_samples)
print("n-features: %d" % n_features)
print("n_classes: %d" % n_classes)


#Split into a training set and a test set using a stratified k fold

#split into a trainging and testing set
#X_train：训练集，X_test：测试集, y_train：训练集label, y_test:测试集label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



#Computing a PCA(eigenfaces) on the face dataset (treated as unlabeled
#dataset): unsupervised feature extraction/dimensionlity reduction
#原始维度太高，先把特征值的维度减小
n_components = 150

print("Extracting the top %d eignfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
#对X_train建模
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

#eigenfaces:人脸上提取的特征值
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfacs orthonormal basis")
t0 = time()
#转化为降维之后的矩阵
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time()-t0))



#Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
#对C和gamma取不同的值
#C：penalty parameter C of the error term
#gamma：if gamma is 0.0 the 1/n_features will be used instead
param_grid = {'C':[1e3,5e3,1e4,1e5],
              'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]}
#GridSearchCV()：将参数进行组合，应用Ssvm算法
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'),param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time()-t0))
print("Best estimator found by grid search: ")
print(clf.best_estimator_)
#Estimator that was chosen by the search, i.e. estimator which gave highest 
#score (or smallest loss if specified) on the left out data. 

#Quantitative evaualtion of hte model quality on the test set

print("Predicting people's names on the best set")
t0 = time()
y_pred = clf.predict(X_test_pca) #对测试集进行预测
print("done in %0.3fs" %(time()-t0))

#对预测结果与真实结果进行比较，再填入真实标签中的姓名
print(classification_report(y_test, y_pred, target_names = target_names))
#建立一个混淆矩阵，可视化预测结果，对角线表示预测对的数
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))


#Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        


#plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_names = target_names[y_pred[i]].rsplit(' ',1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ' ,1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_names, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)


#plot the gallery of the most significations eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

