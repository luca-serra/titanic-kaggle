import numpy as np
from sklearn import model_selection
import pandas as pd
import csv as csv
from sklearn import preprocessing, svm, tree, ensemble, decomposition


# Combining the names' titles into 6 main classes : Civilian (1), Military (2), Intellectual (3), Noble (4)
titles = {'Mr.': 1, 'Mrs.': 1, 'Miss.': 1, 'Master.': 1, 'Don.': 4, 'Rev.': 3, 'Dr.': 3, 'Mme.': 1, 'Ms.': 1, 'Major.': 2, 'Lady.': 4, 'Sir.': 4, 'Mlle.': 1, 'Col.': 2, 'Capt.': 2, 'the': 4, 'Jonkheer.': 4, 'Dona.' : 4}
sex = {'male' : 0, 'female' : 1}
harbors = {'C' : 1, 'Q' : 2, 'S' : 4, '' : 4} # Numeric values corresponding to the harbors (people with an empty harbor field are assigned Southampton, the majority value)
avg_fares = {1 : 84.15, 2 : 20.66, 3 : 13.68}
age_mean = 29.7
age_std = 14.5
number_of_datapoints = 891

# Functions that will be useful later on
def random_age() :
    a = -1
    while a < 0 :
        a = np.random.normal(age_mean, age_std)
    return(a)


def title_vector(title) :
    V = []
    k = titles[title] - 1
    for i in range(max(titles.values())) :
        if i == k :
            V.append(1)
        else :
            V.append(0)
    return V

def pca_decomp(data_matrix, dims_to_remove):
    k = data_matrix.shape[1]
    r = k - dims_to_remove
    pca = decomposition.PCA(n_components=r)
    reduced_data = pca.fit_transform(data_matrix)
    return reduced_data

def svd_decomp(data_matrix, dims_to_remove): # SVD decomposition to reduce dimensionality of the data
    U, S, V = np.linalg.svd(data_matrix)
    S = np.diag(S)
    k = len(S)
    r = k - dims_to_remove
    U = U[:, 0:r]
    S = S[0:r, 0:r]
    V = np.transpose(V)[0:r]
    reduced_matrix = np.dot(U, np.dot(S, V))
    return reduced_matrix


# Load the data
datafile = csv.reader(open('Data/train.csv', 'r'))
next(datafile)
data = [] # Create a variable to hold the data
y = [] # Create a variable to hold the class labels

# Cleaning the data
for row in datafile : # Skip through each row in the csv file
    content = row[0]
    content = content.split(',') # creates a column for each feature (Family Name and Title + First Names are 2 different columns)
    features = [0 for i in range(15)] # will contain the relevant data under numerical form
    namelength = len(content[4])
    content[4] = content[4].split()[0] # only retains the title from the name of the person
    features[14] = namelength
    features[0] = int(content[1]) # first column is the class label
    features[1] = int(content[2]) # second column is the passenger class
    features[2:6] = title_vector(content[4]) # third to sixth columns represent the "title class", obtained by One Hot Encoding
    features[6] = sex[content[5]] # seventh column is the sex (0 for male, 1 for female)
    if content[6] == '' : # if the age field is empty, assigns a pseudo-random age according to the distribution of the dataset
        features[7] = age_mean
        features[8] = 0
        features[13] = 1
    elif len(content[6].split('.')) == 2 and float(content[6]) > 0 : # if the age is estimated, the eigth column (created feature 'estimated age') is assigned +1
        features[7] = float(content[6].split('.')[0])
        features[8] = 1
        features[13] = 0
    else :
        features[7] = float(content[6])
        features[8] = 0
        features[13] = 0
    features[9] = int(content[7]) # the number of siblings/spouses is kept as such
    features[10] = int(content[8]) # the number of parents/children is kept as such
    if content[10] == '' : # if the fare field is empty, assigns the average ticket fare for the corresponding passenger class
        features[11] = avg_fares[features[1]]
    else :
        features[11] = float(content[10])
    features[12] = harbors[content[12]]
    y.append(features[:][0])
    data.append(features[:][1:])

# Creating a dataframe to hold the cleaned data
df = pd.DataFrame(data, columns=['PsngrClass', 'IsCivilian', 'IsMilitary', 'IsIntellectual', 'IsNoble', 'Sex', 'Age', 'AgeIsEstimated', 'Sib/sp', 'Par/ch', 'Fare', 'EmbarkedInSH', 'AgeIsEmpty', 'Name Length'])

df = df.drop(columns=['IsCivilian', 'IsMilitary', 'IsIntellectual', 'IsNoble', 'Name Length'])
X0 = preprocessing.scale(df) # Normalize the numerical data so that it can be processed by classifiers
y = np.array(y) # Convert the labels from a Python list to a numpy array

# Initialize cross validation
kf = model_selection.KFold(n_splits=10)


#%% First classifier : naive sex-based classifier on the original dataset (without dimensionality reduction)

def classify(trainSet, trainLabels, testSet):
	""" gives a prediction according to the sex of the passengers"""
	predictedLabels = np.zeros(testSet.shape[0])
	
	for i in range(testSet.shape[0]):
		if testSet[i,1] > 0 :
			predictedLabels[i] = 1

	return(predictedLabels)

cross_val_sets = kf.split(X0)

totalInstances = 0 # Variable that will store the total instances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted instances  

print('Sex-based classifier :')
for trainIndex, testIndex in cross_val_sets:
    trainSet = X0[trainIndex]
    testSet = X0[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
    
    predictedLabels = classify(trainSet, trainLabels, testSet)#np.zeros(testSet.shape[0])

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print('Accuracy: ' + str(float(correct)/(testLabels.size)))
    totalCorrect += correct
    totalInstances += testLabels.size

print('Total Accuracy of sex-based classiier : ' + str(totalCorrect/float(totalInstances)) + '\n\n')


#%% Second classiier : linear SVM using data with reduced dimensionality (through SVD)

X1 = pca_decomp(X0, 3)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances

C_val = 1
cross_val_sets = kf.split(X1)
clf = svm.LinearSVC(C=C_val)

print('Linear SVM (C = ' + str(C_val) +') :')
for trainIndex, testIndex in cross_val_sets:
    trainSet = X1[trainIndex]
    testSet = X1[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
    clf.fit(trainSet, trainLabels)
    
    predictedLabels = clf.predict(testSet)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print('Accuracy: ' + str(float(correct)/(testLabels.size)))
    totalCorrect += correct
    totalInstances += testLabels.size

print('Total Accuracy of Linear SVM : ' + str(totalCorrect/float(totalInstances)) + '\n\n')


#%% Third classiier : AdaBoost with Decision Trees

X1 = svd_decomp(X0, 3)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances

D = 2
N = 20
cross_val_sets = kf.split(X1)
clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=D), n_estimators=N)

print('Decision Tree with AdaBoost (N = '+ str(N) + ', max depth = '+str(D) +') :')
for trainIndex, testIndex in cross_val_sets:
    trainSet = X1[trainIndex]
    testSet = X1[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
    clf.fit(trainSet, trainLabels)
    
    predictedLabels = clf.predict(testSet)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print('Accuracy: ' + str(float(correct)/(testLabels.size)))
    totalCorrect += correct
    totalInstances += testLabels.size
print('Total Accuracy of AdaBoost  : ' + str(totalCorrect/float(totalInstances)) + '\n\n')


#%% Fourth classifier : decision tree

X1 = svd_decomp(X0, 2)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances

D = 4
cross_val_sets = kf.split(X1)
clf = tree.DecisionTreeClassifier(max_depth=D)

print('Decision Tree (max depth = ' + str(D) + ') :')
for trainIndex, testIndex in cross_val_sets:
    trainSet = X1[trainIndex]
    testSet = X1[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
    clf.fit(trainSet, trainLabels)
    
    predictedLabels = clf.predict(testSet)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print('Accuracy: ' + str(float(correct)/(testLabels.size)))
    totalCorrect += correct
    totalInstances += testLabels.size
print('Total Accuracy of Decision Tree : ' + str(totalCorrect/float(totalInstances)) + '\n\n')

#%% Fifth classiier : Gaussian Kernel SVM

X1 = pca_decomp(X0, 3)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances

C_val = 1.1
gamma_val = 0.25
cross_val_sets = kf.split(X1)
clf = svm.SVC(C=C_val, kernel='rbf', gamma=gamma_val)

print('Gaussian Kernel SVM (C = ' + str(C_val) + ', gamma = ' + str(gamma_val) + ') :')
for trainIndex, testIndex in cross_val_sets:
    trainSet = X1[trainIndex]
    testSet = X1[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
    clf.fit(trainSet, trainLabels)
    
    predictedLabels = clf.predict(testSet)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print('Accuracy: ' + str(float(correct)/(testLabels.size)))
    totalCorrect += correct
    totalInstances += testLabels.size
print('Total Accuracy of Gaussian Kernel SVM : ' + str(totalCorrect/float(totalInstances)) + '\n\n')


#%% Prediction on the actual test dataset

# Load data
test_datafile = csv.reader(open('Data/test.csv', 'r'))
next(test_datafile)
testdata = [] # Create a variable to hold the data
passengersId = [] # List to store the passenger IDs

for row in test_datafile : # Skip through each row in the csv file
    content = row[0]
    content = content.split(',') # creates a column for each feature (Family Name and Title + First Names are 2 different columns)
    features = [0 for i in range(15)] # will contain the relevant data under numerical form
    namelength = len(content[3])
    content[3] = content[3].split()[0] # only retains the title from the name of the person
    features[14] = namelength
    features[0] = int(content[0]) # first column is the ID
    features[1] = int(content[1]) # second column is the passenger class
    features[2:6] = title_vector(content[3]) # third to sixth columns represent the "title class", obtained by One Hot Encoding
    features[6] = sex[content[4]] # seventh column is the sex (0 for male, 1 for female)
    if content[5] == '' : # if the age field is empty, assigns the mean age
        features[7] = age_mean
        features[8] = 0
        features[13] = 1
    elif len(content[5].split('.')) == 2 and float(content[5]) > 0 : # if the age is estimated, the eigth column (created feature 'estimated age') is assigned +1
        features[7] = float(content[5].split('.')[0])
        features[8] = 1
        features[13] = 0
    else :
        features[7] = float(content[5])
        features[8] = 0
        features[13] = 0
    features[9] = int(content[6]) # the number of siblings/spouses is kept as such
    features[10] = int(content[7]) # the number of parents/children is kept as such
    if content[9] == '' : # if the fare field is empty, assigns the average ticket fare for the corresponding passenger class
        features[11] = avg_fares[features[1]]
    else :
        features[11] = float(content[9])
    features[12] = harbors[content[11]]
    passengersId.append(features[:][0])
    testdata.append(features[:][1:])

df_test = pd.DataFrame(testdata, columns=['PsngrClass', 'IsCivilian', 'IsMilitary', 'IsIntellectual', 'IsNoble', 'Sex', 'Age', 'AgeIsEstimated', 'Sib/sp', 'Par/ch', 'Fare', 'EmbarkHarbor', 'AgeIsEmpty', 'Name Length'])

df_test = df_test.drop(columns=['IsCivilian', 'IsMilitary', 'IsIntellectual', 'IsNoble', 'Name Length'])
X0_test = preprocessing.scale(df_test) # Normalize the numerical data so that it can be processed by classifiers
print(df_test.head())

X1 = pca_decomp(X0, 3)
X1_test = pca_decomp(X0_test, 3)

C_val = 1
clf = svm.LinearSVC(C=C_val)

clf.fit(X1, y)
predicted_labels_test = clf.predict(X1_test)

test_size = len(predicted_labels_test)
formatted_results = [(str(passengersId[i]) + ',' + str(int(predicted_labels_test[i])) + '\n') for i in range(test_size)]
with open('Predictions/predicted_class_LinearSVM3.csv', 'w') as myfile:
    myfile.write('PassengerId,Survived\n')
    for i in range(test_size) :
        myfile.write(formatted_results[i])

nb_dead_test = 0
for l in predicted_labels_test :
    if l == 0 :
        nb_dead_test += 1

print('Predicted death rate :')
print(nb_dead_test/test_size)

