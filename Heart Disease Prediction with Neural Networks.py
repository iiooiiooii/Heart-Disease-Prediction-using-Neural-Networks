import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# import the heart disease dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# the names will be the names of each column in our pandas DataFrame
names = ['age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
        'class']

# read the csv
cleveland = pd.read_csv(url, names=names)


# print the shape of the DataFrame, so we can see how many examples we have
print ('Shape of DataFrame: {}'.format(cleveland.shape))
print (cleveland.loc[1])


# print the shape of the DataFrame, so we can see how many examples we have
print ('Shape of DataFrame: {}'.format(cleveland.shape))
print (cleveland.loc[1])


# remove missing data (indicated with a "?")
data = cleveland[~cleveland.isin(['?'])]
data.loc[280:]

# drop rows with NaN values from DataFrame
data = data.dropna(axis=0)
data.loc[280:]

# print the shape and data type of the dataframe
print (data.shape)
print (data.dtypes)

# transform data to numeric to enable further analysis
data = data.apply(pd.to_numeric)
data.dtypes

# print data characteristics, usings pandas built-in describe() function
data.describe

# plot histograms for each variable
data.hist(figsize = (12, 12))
plt.show()

# create X and Y datasets for training
from sklearn import model_selection

X = np.array(data.drop(['class'], 1))
y = np.array(data['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

# convert the data to categorical labels
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])
