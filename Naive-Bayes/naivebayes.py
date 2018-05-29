# -*- coding: utf-8 -*-
import numpy as np
 
 #词汇表到向量的转换函数
def loadDataSet():
     postingList = [['my','dog','has','flea','problems','help','please'],
                    ['maybe','not','take','him','to','dog','park','stupid'],
                    ['my','dalmation','is','so','cute','I','love','him'],
                    ['stop','posting','stupid','worthless','garbage'],
                    ['mr','licks','ate','my','steak','how','to','stop','him'],
                    ['quit','buying','worthless','dog','food','stupid']]
     classVec = [0,1,0,1,0,1] #1代表侮辱性文字，0代表正常言论
     return postingList, classVec
 
#创建一个空集
def createVocabList(dataSet):
    vocabSet = set([]) #使用set创建不重复词表库
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建不重复的单词向量,vocabSet为所有不重复单词
    return list(vocabSet)

#创建一个其中所含元素都为0的向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList) #创建一个包含所有元素都为0的向量
    #遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word： %s is not in my Vocabulary!" % word
    return returnVec #对每个句子二值化，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    """朴素贝叶斯分类器训练函数。trainMatrix:文档矩阵，trainCategory:每篇文档类别标签"""
    numTrainDocs = len(trainMatrix) #6
    numWords = len(trainMatrix[0]) #32
    pAbusive = sum(trainCategory)/float(numTrainDocs) #侮辱性标签所占比例
    #初始化所有词出现数为1，并将分母初始化为2，避免某一个概率值为0
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)  #change to ones()
    p0Denom = 2.0; p1Denom = 2.0  #change to 2.0 
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i]) #所有元素求和
    ##将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p1Vect = np.log(p1Num/p1Denom)  #change to ln()
    p0Vect = np.log(p0Num/p0Denom)  #每个特征在该类别下的概率
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
#    print "p1:", p1
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
#    print "p0:", p0
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))    
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    

def main():
    testingNB()
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    