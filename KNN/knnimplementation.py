# -*- coding: utf-8 -*-

import csv
import random
import math
import operator

#装载数据
def loadDataset(filename,split,trainingSet=[],testSet=[]): #split用于选择一个阈值将原始数据分为
    with open(filename,'rb') as csvfile:                   #trainingSet和testsET
        lines = csv.reader(csvfile)
        dataset = list(lines)   #将行转化为list的数据类型
        for x in range(len(dataset)-1):
            for y in range (4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

#计算距离 可能是多维  length表示维度         
def euclideanDistance(instance1,instance2,length):
    distance = 0
    for x in range(length): #对每一维度
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

#返回最近的K个邻居
def getNeighbors(trainingSet,testInstance,k):
    distance = []
    length = len(testInstance)-1  #要测试的实例维度
    for x in range(len(trainingSet)): #对训练集的每一个实例进行计算
        dist = euclideanDistance(testInstance,trainingSet[x],length)
        distance.append((trainingSet[x],dist))
    distance.sort(key = operator.itemgetter(1)) #对距离从小到大排序
    neighbors = []
    for x in range(k):  #取前K个距离
        neighbors.append(distance[x][0])
    return neighbors

#利用返回的最近的K个点的类型，根据少数服从多数的原则进行分类
def getResponse(neighbors): 
    classVotes = {}
    for x in range(len(neighbors)):  #统计每一个分类中邻居的个数
        response = neighbors[x][-1] #[-1]意思最后一个值，即分类的label
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #把每一个类投票的个数降序方式排列
    sortedVotes = sorted(classVotes.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedVotes[0][0] #返回投票最多的分类类别

#测算估计的值与实际的值相比的准确率    
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    #preparedata
    trainingSet = []
    testSet = []
    split = 0.67  #2/3数据划分为训练集
    loadDataset(r'F:\pythonpractice\data\irisdata.txt',split,trainingSet,testSet)
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))
    #generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet,testSet[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted = ' + repr(result) + ', actual = ' + repr(testSet[x][-1]))
    print('predictions: ' + repr(predictions))
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
if __name__ == '__main__':
    main()

