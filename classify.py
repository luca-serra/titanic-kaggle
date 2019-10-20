from numpy import *

def classify(trainSet, trainLabels, testSet):
	
	predictedLabels = zeros(testSet.shape[0])
	
	for i in range(testSet.shape[0]):
		if testSet[i,1] > 0 :
			predictedLabels[i] = 1

	return predictedLabels