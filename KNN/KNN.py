#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  KNN-algo.py
#  
# #-*-coding:utf-8-*-
from numpy import *
import operator
from os import listdir
#inX表示新测试数据，dataSet表示训练集，labels训练集对应的数字，k表示采用几近邻算法
def classify0(inX, dataSet, labels, k):
	# 获取数据的行数，shape[1]为列数
    dataSetSize = dataSet.shape[0]
    '''
    tile(A,(m,n))   
    print (dataSet)
    print ("----------------")
    #将矩阵扩展，行数要和训练集相同，也就是100行
    print (tile(inX, (dataSetSize,1)))
    print ("----------------")
    '''
    #求两个矩阵的差
    diffMat = tile(inX, (dataSetSize,1)) - dataSet      
    #print (diffMat)
    #求差的平方，是对矩阵的每一个元素平方
    sqDiffMat = diffMat**2
    #对每一行求和，axis=0表示每列相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，，算出各个距离        
    distances = sqDistances**0.5
    #print(distances)
    #对距离进行排序,argsort函数:返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()          
    classCount={}                                      
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #这里的0表示如果这个voteIlabel不存在时，其值取零
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #classCount是一个字典，根据第二个字段进行降序排序，获得一个以元组为元素的列表
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
	#取这个列表中的第一个元组，取这个元组的第0相，就是我们要的label
    return sortedClassCount[0][0]

def img2vector(filename):
	#初始化一个1*1024的零数组用于存储一个32*32的图片，一张图片就对应了一个特征向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
		#读取文件的一行
        lineStr = fr.readline()
        #把这一行的每一个字符存入数组对应的位置
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    #获取训练集的文件目录
    trainingFileList = listdir('trainingDigits')          
    lengthTrainingFileList = len(trainingFileList)
    #根据训练集的文件个数构建一个对应行数，列数为1024的零矩阵
    trainingMat = zeros((lengthTrainingFileList,1024))
    for i in range(lengthTrainingFileList):
		#获取第i个文件的文件名
        fileNameStr = trainingFileList[i]
        #获取去掉后缀的文件名         
        fileStr = fileNameStr.split('.')[0]
        #获取文件表示的数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #print hwLabels
        #print fileNameStr
        #把这个文件对应的32*32矩阵放到对应的那行   
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
        #print trainingMat[i,:]
        #print len(trainingMat[i,:])
	#到这里，所有的文件都转化好了，我们获得了一个100*1024的矩阵，每行是一个手写数字的特征矩阵
	#还有一个hwLabels对应了某行表示的数字
	#以下对测试文件做同样处理
    testFileList = listdir('testDigits')       
    errorCount = 0.0
    lengthTestFileList = len(testFileList)
    for i in range(lengthTestFileList):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #利用上面定义的分类函数对测试集进行分类，查看错误率
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 5)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(lengthTestFileList)))



def main(args):
    return 0

if __name__ == '__main__':
    import sys
    handwritingClassTest()
    sys.exit(main(sys.argv))
