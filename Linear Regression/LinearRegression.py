import numpy as np
import math

##------------------##
#建立训练数据和测试数据#
##------------------##
def loadDataSet(dataSet):
	data = np.loadtxt(dataSet)
	#建立测试集和训练集的索引，在数据集中随机取30%的数据作为测试数据
	shuffle_indexes = np.random.permutation(len(data))
	test_ratio = 0.3
	test_size = int(len(data) * test_ratio)
	test_indexes = shuffle_indexes[:test_size]
	train_indexes = shuffle_indexes[test_size:]
	#取出训练集和测试结，将第一列的序号改成截距项，最后一列为因变量
	trainX = data[train_indexes]
	trainX[:,0] = 1
	trainY = np.copy(trainX[:,-1])
	testX = data[test_indexes]
	testX[:,0] = 1
	testY = np.copy(testX[:,-1])
	#在X中将因变量删除,第三个变量为axis，1表示列，0表示行，none表示扁平化
	trainX = np.delete(trainX,-1,1)
	testX = np.delete(testX,-1,1)
	return trainX,trainY,testX,testY

##------------------##
#计算参数beta#
##------------------##
def betaHat(trainX,trainY):
	#betaHat = np.dot(np.dot(np.linalg.inv(np.dot(trainX.transpose(),trainX)),trainX.transpose()),trainY)
	trainXt = np.transpose(trainX)
	trainXttrainX = np.dot(trainXt,trainX)
	trainXttrainY = np.dot(trainXt,trainY)
	betaHat = np.linalg.solve(trainXttrainX,trainXttrainY)
	return betaHat

##------------------##
#获取测试集上的预测值#
##------------------##	
def predict(newData,beta):
	predictValue = np.dot(newData,beta)
	return predictValue

##------------------##
#计算均方误差#
##------------------##
def evaluation(testY,predictValue):
	rss = np.dot((testY-predictValue).transpose(),testY-predictValue)
	return rss
	
if __name__ == '__main__':
	trainX,trainY,testX,testY = loadDataSet("prostate.csv")
	beta = betaHat(trainX,trainY)
	predictValue = predict(testX,beta)
	rss = evaluation(testY,predictValue)
	print("real Y values : \n",testY)
	print("\npredict Y values : \n",predictValue)
	print("\nRSS = ",rss)
