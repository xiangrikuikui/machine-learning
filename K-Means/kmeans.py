# -*- coding: utf-8 -*-

import numpy as np


def kmeans(X, k, maxIt):
    numPoints, numDim = X.shape  #numPoints行，numDim列
    
    dataSet = np.zeros((numPoints, numDim + 1))  #多加1列
    dataSet[:, :-1] = X  #不包括倒数第一列

    # Initialize centroids randomly
    # 随机从numPoints行中选出k行
    centroids = dataSet[np.random.randint(numPoints, size = k), :]
    centroids = dataSet[0:2, :]
    # Randomly assign labels to initial centroid
    centroids[:, -1] = range(1, k+1)  #将中心点设为1到k类

    # Initialize book keeping vars
    iterations = 0
    oldCentroids = None
    
    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print "iterations: \n", iterations
        print "dataSet: \n", dataSet
        print "centroids: \n", centroids
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = np.copy(centroids)
        iterations += 1
        
        # Assign labels to each datapoint based on centroids
        updateLabels(dataSet, centroids)
        
        # Assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, k)
        
    # We can get the labels too by calling getLabels(dataSet, k)
    return dataSet




def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
     #值是否相等；相等的话就结束，false的话就继续
    return np.array_equal(oldCentroids, centroids) 
 
def updateLabels(dataSet, centroids):
    numPoints, numDim = dataSet.shape
    for i in range(0, numPoints):
        # 对比当前行和中心点的距离，把最近的中心点的label返回到当前行的最后一列
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)


def getLabelFromClosestCentroid(dataSetRow, centroids):
    label = centroids[0,-1]  #让label先等于第一个中心点的值
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1, centroids.shape[0]): #shape[0]表示行
        # np.linalg.norm计算距离
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]
    print "minDist: ", minDist
    return label


# 重新求中心点坐标
def getCentroids(dataSet, k):
    
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):  #1-k
        # 将label等于i的所有行的从开始到倒数第二列都取出来 
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        # 行坐标从0开始，中心点坐标从1开始，所以-1
        # 求均值，赋给中心点的所有列，不包括最后一列
        result[i - 1, :-1] = np.mean(oneCluster, axis = 0) #axis = 0:对所有行的每一列求均值
        # 重赋标签 标签值未变
        result[i - 1, -1] = i
        
    return result

x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
testX = np.vstack((x1, x2, x3, x4))   #将四个点纵向堆叠起来构成矩阵

result = kmeans(testX, 2, 10)
print "final result:"
print result
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    