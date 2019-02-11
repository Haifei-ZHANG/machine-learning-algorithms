import numpy as np

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
	trainY = np.copy(trainX[:,-1])
	testX = data[test_indexes]
	testY = np.copy(testX[:,-1])
	#在X中将因变量删除,第三个变量为axis，1表示列，0表示行，none表示扁平化
	trainX = np.delete(trainX,-1,1)
	testX = np.delete(testX,-1,1)
	return data,trainX,trainY,testX,testY

def parametersEstimator(trainX,trainY):
	#pi1表示spam的先验概率，pi0表示non spam的先验概率
	pi1 = np.count_nonzero(trainY)/len(trainY)
	pi0 = 1-pi1
	#mu1表示spam的均值，mu0表示nom spam的均值，均为长度为57的向量
	mu1 = np.mean(trainX[trainY==1,:],axis=0)
	mu0 = np.mean(trainX[trainY==0,:],axis=0)
	#LDA中我们假设两个类别的协方差矩阵是相等的
	sigma = np.cov(trainX.T)
	return pi1,pi0,mu1,mu0,sigma
	
def thresholdValue(pi1,pi0,mu1,mu0,sigma):
	return 0.5*np.dot(np.dot(mu1+mu0,np.linalg.inv(sigma)),mu1-mu0)-np.log(pi1/pi0)
	

def predict(testX,mu1,mu0,sigma,s):
	yHat = np.dot(np.dot(testX,np.linalg.inv(sigma)),mu1-mu0)
	spam = yHat>=s
	yHat[spam]=1
	yHat[~spam]=0
	return yHat

def evaluation(testY,yHat):
	table = np.zeros((2,2))
	for i in range(0,len(testY)):
		if(testY[i]==yHat[i]):
			if(testY[i]==1):
				table[0,0] += 1
			else:
				table[1,1] += 1
		else:
			if(testY[i]==1):
				table[0,1] += 1
			else:
				table[1,0] += 1
	print("classification table : \n  TP  |  FN\n------------\n  FP  |  TN \n\n",table)
	print("error rate : ",1-np.sum(testY==yHat)/len(testY))
	
def run(dataSet):
	data,trainX,trainY,testX,testY = loadDataSet(dataSet)
	pi1,pi0,mu1,mu0,sigma = parametersEstimator(trainX,trainY)
	s = thresholdValue(pi1,pi0,mu1,mu0,sigma)
	yHat = predict(testX,mu1,mu0,sigma,s)
	evaluation(testY,yHat)
	
if __name__ == '__main__':
	print("begin\n")
	run("spambase.dat")
