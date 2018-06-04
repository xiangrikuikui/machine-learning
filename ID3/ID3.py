# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 14:11:43 2018

@author: lenovo
"""

from numpy import *
#load 'iris.data' into our workspace
traindata = loadtxt(".\\Iris.txt", delimiter = ',', usecols = (0,1,2,3), dtype = float)
trainlabel = loadtxt(".\\Iris.txt", delimiter = ',', usecols = (range(4,5)), dtype = str)
feaname = ["#0", "#1", "#2", "#3"] #redefine feature names

#calculate entropy
from math import log
def calentropy(label):
    n = label.size
    count = {}
    for curlabel in label:
        if curlabel not in count.keys():  #count.keys()返回count中所有的键
            count[curlabel] = 0
        count[curlabel] += 1 
    entropy = 0
    for key in count:
        pxi = float(count[key])/n
        entropy -= pxi * log(pxi,2)
    return entropy

args = mean(traindata, axis=0) #将矩阵按列求均值，返回1*n的矩阵 args = [0,0,0,0]
#args = [5.84333333, 3.054, 3.75866667, 1.19866667]

#split traindata according to splitfea_idx
def splitdata(oridata, splitfea_idx): #对连续型数据每一个特征按照特征均值分为两类
    arg = args[splitfea_idx]
    idx_less = []
    idx_greater = []
    n = len(oridata)
    for idx in range(n):
        d = oridata[idx]  #d is a list 
        if d[splitfea_idx] < arg:
            idx_less.append(idx)
        else:
            idx_greater.append(idx)
    splitidx = [idx_less, idx_greater]
    return splitidx #对每个特征按照小于均值，大于均值的顺序排列,第一行全小于均值，第二行大于均值
# idx_less, idx_greater = splitdata(traindata, 2)
# splitidx = [idx_less, idx_greater] #splited labels

#transform splitidx into splited traindata and label
def idx2data(oridata,label,splitidx,fea_idx):
    idx1 = splitidx[0]
    idx2 = splitidx[1]
    datal = []
    datag = []
    labell = []
    labelg = []
    for i in idx1:
        datal.append(append(oridata[i][:fea_idx],oridata[i][fea_idx+1:]))
    for i in idx2:
        datag.append(append(oridata[i][:fea_idx],oridata[i][fea_idx+1:]))
    labell = label[idx1]
    labelg = label[idx2]
    return datal,datag,labell,labelg

#choose the best fea_idx
def choosebest_splitnode(oridata,label):
    n_fea = len(oridata[0])
    n = len(label)
    base_entropy = calentropy(label)
    best_gain = -1
    for fea_i in range(n_fea):
        cur_entropy = 0
        idxset_less, idxset_greater = splitdata(oridata,fea_i)
        prob_less = float(len(idxset_less))/n
        prob_greater = float(len(idxset_greater))/n
        
        cur_entropy += prob_less * calentropy(label[idxset_less])
        cur_entropy += prob_greater * calentropy(label[idxset_greater])
        
        info_gain = base_entropy - cur_entropy;
        if info_gain > best_gain:
            best_gain = info_gain
            best_idx = fea_i
    return best_idx

#build tree
def buildtree(oridata,label):
    #label id null
    if label.size == 0:
        return "NULL"
    #the label is same
    listlabel = label.tolist()
    if listlabel.count(label[0]) == label.size:
        return label[0]
    #all the features are used
    if len(feaname) == 0:
        cnt = {}
        for cnt_l in label:
            if cnt_l not in cnt.keys():
                cnt[cnt_l] = 0
            cnt[cnt_l] += 1
        maxx = -1
        for keys in cnt:
            if cnt[keys] > maxx:
                maxx = cnt[keys]
                maxkey = keys
        return maxkey
    
    bestsplit_fea = choosebest_splitnode(oridata,label)
    print(bestsplit_fea, len(oridata[0]))
    cur_feaname = feaname[bestsplit_fea]
    print(cur_feaname)
    nodedict = {cur_feaname:{}}
    del(feaname[bestsplit_fea])
    split_idx = splitdata(oridata, bestsplit_fea)
    data_less,data_greater,label_less,label_greater = idx2data(oridata,label,split_idx,bestsplit_fea)
    
    nodedict[cur_feaname]["<"] = buildtree(data_less, label_less)
    nodedict[cur_feaname][">"] = buildtree(data_greater, label_greater)
    return nodedict
#classify for the test tree
feanamecopy = ["#0","#1","#2","#3"] #re-defined feature names
def classify(mytree, testdata):
    if type(mytree).__name__ != 'dict':
        return mytree
    fea_name = list(mytree.keys())[0]
    fea_idx = feanamecopy.index(fea_name)    
    val = testdata[fea_idx]
    nextbranch = mytree[fea_name]
    
    if val > args[fea_idx]:
        nextbranch = nextbranch[">"]
    else:
        nextbranch = nextbranch["<"]
    return classify(nextbranch,testdata)

#test
mytree = buildtree(traindata,trainlabel)
tt = traindata[0]
tt_label = classify(mytree, tt)
print(tt_label)














