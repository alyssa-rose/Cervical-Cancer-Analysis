import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data = pd.read_csv("cervicalCancer.csv")
# removing primarily useless columns
data = data.drop(data.columns[[14,21,26,27]], axis = 1)

# converting the '?' in original data to NAN and
# removing
data = data.replace('?', np.nan)
data = data.dropna(axis = 0)

# saving the cleaned file to use in MATLAB
data.to_csv(r'C:\Users\alyss\OneDrive\Documents\College\Math123\Final Paper\cervicalCancerFix.csv')
# classification by Artificial Neural Networks


# the 4 possible targets
Hinselmann = data.iloc[:, 28].values
Schiller = data.iloc[:, 29].values
Cytology = data.iloc[:, 30].values
Biopsy = data.iloc[:, 31].values

X = data.iloc[:, 0:28].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Hinselmann, test_size = 0.25, random_state = 0)

from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Schiller, test_size = 0.25, random_state = 0)

from sklearn.model_selection import train_test_split
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Cytology, test_size = 0.25, random_state = 0)

from sklearn.model_selection import train_test_split
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, Biopsy, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train3 = sc.fit_transform(X_train3)
X_test3 = sc.transform(X_test3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train4 = sc.fit_transform(X_train4)
X_test4 = sc.transform(X_test4)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

"""
HINSELMANN CLASSIFICATION
"""
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(kernel_initializer="uniform", input_dim=28, units=11, activation="relu"))

# Adding the second hidden layer
classifier.add(Dense(kernel_initializer="uniform", units=11, activation="relu"))

# Adding the output layer
classifier.add(Dense(kernel_initializer="uniform", units=1, activation="sigmoid"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 250)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


"""
SCHILLER CLASSIFICATION
"""
classifier2 = Sequential()

# Adding the input layer and the first hidden layer
classifier2.add(Dense(kernel_initializer="uniform", input_dim=28, units=11, activation="relu"))

# Adding the second hidden layer
classifier2.add(Dense(kernel_initializer="uniform", units=11, activation="relu"))

# Adding the output layer
classifier2.add(Dense(kernel_initializer="uniform", units=1, activation="sigmoid"))

# Compiling the ANN
classifier2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier2.fit(X_train2, y_train2, batch_size = 10, epochs = 250)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred2 = classifier2.predict(X_test2)
y_pred2 = (y_pred2 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test2, y_pred2)


"""
CYTOLOGY CLASSIFICATION
"""
classifier3 = Sequential()

# Adding the input layer and the first hidden layer
classifier3.add(Dense(kernel_initializer="uniform", input_dim=28, units=11, activation="relu"))

# Adding the second hidden layer
classifier3.add(Dense(kernel_initializer="uniform", units=11, activation="relu"))

# Adding the output layer
classifier3.add(Dense(kernel_initializer="uniform", units=1, activation="sigmoid"))

# Compiling the ANN
classifier3.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier3.fit(X_train3, y_train3, batch_size = 10, epochs = 250)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred3 = classifier3.predict(X_test3)
y_pred3 = (y_pred3 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test3, y_pred3)

"""
BIOPSY
"""
classifier4 = Sequential()

# Adding the input layer and the first hidden layer
classifier4.add(Dense(45, input_dim=28, activation="relu"))

# Adding the second hidden layer
classifier4.add(Dense(28, activation="relu"))

# Adding the output layer
classifier4.add(Dense(1, activation="sigmoid"))

# Compiling the ANN
classifier4.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier4.fit(X_train4, y_train4, batch_size = 10, epochs = 250)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred4 = classifier4.predict(X_test4)
y_pred4 = (y_pred4 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test4, y_pred4)


"""
CLASSIFICATION VIA A SUPPORT VECTOR MACHINE
"""
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

classifier2 = SVC(kernel = 'rbf', random_state = 0)
classifier2.fit(X_train2, y_train2)

classifier3 = SVC(kernel = 'rbf', random_state = 0)
classifier3.fit(X_train3, y_train3)

classifier4 = SVC(kernel = 'rbf', random_state = 0)
classifier4.fit(X_train4, y_train4)

y_predSVMKernel = classifier.predict(X_test)
y_predSVM2Kernel = classifier2.predict(X_test2)
y_predSVM3Kernel = classifier3.predict(X_test3)
y_predSVM4Kernel = classifier4.predict(X_test4)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmSVMKernel = confusion_matrix(y_test, y_predSVMKernel)
cmSVM2Kernel = confusion_matrix(y_test2, y_predSVM2Kernel)
cmSVM3Kernel = confusion_matrix(y_test3, y_predSVM3Kernel)
cmSVM4Kernel = confusion_matrix(y_test4, y_predSVM4Kernel)

# doing normal SVM
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

classifier2 = SVC(kernel = 'linear', random_state = 0)
classifier2.fit(X_train2, y_train2)

classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train3, y_train3)

classifier4 = SVC(kernel = 'linear', random_state = 0)
classifier4.fit(X_train4, y_train4)

y_predSVM = classifier.predict(X_test)
y_predSVM2 = classifier2.predict(X_test2)
y_predSVM3 = classifier3.predict(X_test3)
y_predSVM4 = classifier4.predict(X_test4)

cmSVM = confusion_matrix(y_test, y_predSVM)
cmSVM2 = confusion_matrix(y_test2, y_predSVM2)
cmSVM3 = confusion_matrix(y_test3, y_predSVM3)
cmSVM4 = confusion_matrix(y_test4, y_predSVM4)
