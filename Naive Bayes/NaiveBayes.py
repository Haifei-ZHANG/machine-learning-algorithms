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
	sigma1 = np.diag(np.cov(trainX[trainY==1,:].T))
	sigma0= np.diag(np.cov(trainX[trainY==0,:].T))
	return pi1,pi0,mu1,mu0,sigma1,sigma0

def normalProb(x,mu,sigma):
	prob = (1/np.sqrt(2*3.1415926*sigma))*np.exp(-np.square(x-mu)/(2*sigma))
	return prob
	
def predict(testX,pi1,pi0,mu1,mu0,sigma1,sigma0):
	nTest = len(testX)
	nPredictor = len(mu1)
	yHat = np.ones(nTest)
	for i in range(0,nTest):
		p1 = pi1
		p0 = pi0
		for j in range(0,nPredictor):
			p1 = p1*normalProb(testX[i][j],mu1[j],sigma1[j])
			p0 = p0*normalProb(testX[i][j],mu0[j],sigma0[j])
		if(p1>p0):
			yHat[i] = 1
		else:
			yHat[i] = 0
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

def main(dataSet):
	data,trainX,trainY,testX,testY = loadDataSet(dataSet)
	pi1,pi0,mu1,mu0,sigma1,sigma0 = parametersEstimator(trainX,trainY)
	yHat = predict(testX,pi1,pi0,mu1,mu0,sigma1,sigma0)
	evaluation(testY,yHat)

if __name__ == '__main__':
	print("begin\n")
	main("spambase.dat")
