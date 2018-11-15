import numpy as np
import math
import copy

##------------------##
#建立训练数据和测试数据#
##------------------##
def loadDataSet(dataFileTrain,dataFileTest):
	trainX = []
	trainY = []
	testX = []
	testY = []
	train = np.mat(np.loadtxt(dataFileTrain))	#读入数据，直接转换成矩阵
	trainY = copy.deepcopy(train[:,-1].T)	#提取最后一列的y，转置成一个行向量
	train[:,-1] = 1	#最后一列全为1作为截距
	trainX = train.T	#每一列是一个样本，这样写起来更加方便，之后的参数beta
	#print(trainY)
	test = np.mat(np.loadtxt(dataFileTest))
	testY = copy.deepcopy(test[:,-1].T)
	test[:,-1] = 1
	testX = test.T
	return trainX,trainY,testX,testY

##-----------------------------------##
#sigmod函数，将一个实数转换成（0,1）的实数
##-----------------------------------##
def sigmoid(x):
    return 1.0 / (1+math.exp(-x))

##		  				##
#通过牛顿法去获取最佳的参数w
##		 				##
def newtonMethod(X, Y, iterNum=50):
	m = X.shape[1]	#样本的数量，在我们用的数据集中，m=299
	n = X.shape[0]	#样本的维数，在我们用的数据集中，n=21
	beta = np.mat([0.0] * n).T	#beta是一个列向量
	dBeta = np.mat([0.0] * n).T
	d2Beta = np.mat(np.zeros([n,n]))
	p1 = np.mat([0.0] * m)
	for k in range(iterNum):
		#计算p1,在已知x和beta的情况下，x被判别为1的概率
		z = beta.T*X
		for i in range(m):
			p1[0,i] = sigmoid(z[0,i])
		#计算beta的一阶导数dBeta
		dBeta = -X*((Y-p1).T)
		#计算beta的二阶导数d2beta
		for i in range(m):
			d2Beta = d2Beta+X[:,i]*X[:,i].T*p1[0,i]*(1-p1[0,i])
		beta = beta-d2Beta.I*dBeta	 
	return beta

##		  				      ##
#利用测试集对训练得到的模型进行评估
##		 				      ##
def test(X,Y,beta):
	zHat = beta.T*X
	mTest = Y.shape[1]
	yHat = np.mat([0] * mTest)
	error = 0.0
	for i in range(mTest):
		if sigmoid(zHat[0,i]) >= 0.5 :
			yHat[0,i] = 1
		else:
			yHat[0,i] = 0
		error += np.square(yHat[0,i]-Y[0,i])
	error = error/mTest	
	return error,yHat

trainX,trainY,testX,testY = loadDataSet("horseColicTraining.txt","horseColicTest.txt")
beta = newtonMethod(trainX, trainY)
error, yHat = test(testX,testY,beta)
print(beta)
print(testY)
print(yHat)
print(error)	
