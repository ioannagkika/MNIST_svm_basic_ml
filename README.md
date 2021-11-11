# MNIST_svm_basic_ml
Training a basic ML model using MNIST dataset. Contains dimensionality reduction with PCA, best model selection with grid search and cross validation and comparison with kNN.

In this project the MNIST dataset is used. It can be found on: https://www.kaggle.com/oddrationale/mnist-in-csv
Please make sure the 2 csv files have been downloaded and saved with the names mnist_train.csv and mnist_test.csv. 

Python 3.7.3 was used.

Installing a few python packages is required. The easiest way to install is using pip (or pip3 if python2.x and 3.x are both installed) with the following command:
```
pip install -r requirements.txt
```
Consider using a virtual environment to avoid future version conflicts.

Runs with: 
```
python mnist.py
```
Replace python with python3 if needed.

The ```mnist.py``` script will train an SVM using a subset of the MNIST dataset.

First of all, MNIST has 10 classes, the labels of handwritten digit images. We convert the labels to two classes, 'even' if the digit in question is an even number and 'odd' otherwise. 

Secondly, we only consider the first 6000 rows of our train data and the first 2000 rows of our test set. SVMs are notoriously RAM consuming so it is unrealistic to think that training with 60k rows from the entire dataset is feasible for most users.

The script will first normalize the input using a standard scaler, which is common practice in ML. It will then use PCA to drop the dimensionality from 784 to 200 and subsequently run a grid search to properly parametrize the SVM. This means that it will fit many SVMs in our train data and evaluate them using the Cross Validation method without having access to our test data. Then the SVM with the best train set performance is the one which will be used for our test set. The script automatically chooses the one who performed the best in terms of classification accuracy.

Lastly, additional comparisons are made with other classification algorithms. I've included k-Nearest Neighbor for different k values and Nearest Centroid. Metrics will be printed after every fitting for both train and test.

