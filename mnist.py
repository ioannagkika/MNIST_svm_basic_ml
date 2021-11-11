# Importing the libraries needed

from sklearn import svm, preprocessing, neighbors, model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import time
import numpy as np

# Importing MNIST dataset

traindata = pd.read_csv("./mnist_train.csv")
testdata = pd.read_csv("./mnist_test.csv")

# Separating them to x_train, y_train, x_test, y_test and to odd and even

X_train = traindata.iloc[:6000,1:]
y_train_1 = traindata.iloc[:6000,0]
X_test = testdata.iloc[:4000,1:]
y_test_1 = testdata.iloc[:4000,0]

y_train_1 = np.array(y_train_1)
y_train = []

for i in range(0, len(y_train_1)):
    if y_train_1[i] % 2 == 0:
        y_train.append("even")
    else:
        y_train.append("odd")

y_test_1 = np.array(y_test_1)
y_test = []

for i in range(0, len(y_test_1)):
    if y_test_1[i] % 2 == 0:
        y_test.append("even")
    else:
        y_test.append("odd")
        
#Scaling X_train and X_test
        
standardscaler = preprocessing.StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)

#PCA

pca = PCA(n_components=200)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(sum(pca.explained_variance_ratio_))


# Initializing the cross validated grid search with the following parameters

parameters = [{'kernel': ('sigmoid', 'rbf'), 'gamma': [0.01, 0.1, 1, 1.5, 10],'C': [1, 10, 50, 100]},
              {'kernel': ['linear'], 'C': [1, 10, 50], 'gamma': [0.1, 1, 1.5]},
              {'kernel': ['poly'], 'C': [1, 10, 50], 'degree': [2,3,4,5],'gamma': [0.1, 1, 1.5]}
              ]
clf = model_selection.GridSearchCV(svm.SVC(), parameters, cv = 3, scoring = 'accuracy', verbose = 4)
clf.fit(X_train, y_train)


#Printing the best score and the corresponding parameters of the grid search

print(clf.best_score_)
print(clf.best_params_)
print()

# Printing results for parameters' combinations

score = clf.cv_results_['mean_test_score']
for sc, params in zip(score, clf.cv_results_['params']):
    print("%0.3f for %r"% (sc, params))
    print()


# Initializing an SVM model with the best parameters

svm = svm.SVC(kernel= clf.best_params_.get('kernel'), C = clf.best_params_.get('C'), gamma = clf.best_params_.get('gamma'), degree = clf.best_params_.get('degree'))

# Will keep an eye on training time as well

t1_start = time.process_time()
svm.fit(X_train, y_train)

# End of fitting

t1_stop = time.process_time()


# Predictions of X_test and X_train

y_pred1 = svm.predict(X_test)
y_pred2 = svm.predict(X_train)

# Model evaluation in test and train sets

print("Model evaluation in test")
print()
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

print("Model evaluation in train")
print()
print(confusion_matrix(y_train, y_pred2))
print(classification_report(y_train, y_pred2))
print("Time process: ", t1_stop - t1_start)

# k - Neighbors model

k = 4

for n in range(1,k):
    print("Neighbors: ", n)
    model2 = neighbors.KNeighborsClassifier(n_neighbors=n, weights ='distance', metric='euclidean')
    t1_start = time.process_time()
    model2.fit(X_train, y_train)
    t1_stop = time.process_time()
    y_predicted = model2.predict(X_test)
    # Printing metrics
    print(confusion_matrix(y_test,y_predicted))
    print(classification_report(y_test,y_predicted))
    print("Time process: ", t1_stop-t1_start)
    
#Nearest Centroid model
    
nearestcentroid = neighbors.nearest_centroid.NearestCentroid()
t1_start = time.process_time()
nearestcentroid.fit(X_train, y_train)
t1_stop = time.process_time()
y_pr = nearestcentroid.predict(X_test)
# Printing metrics
print(confusion_matrix(y_test,y_pr))
print(classification_report(y_test,y_pr))
print("Time process: ", t1_stop-t1_start)



