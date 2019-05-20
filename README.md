# Cervical-Cancer-Analysis

Analysis of a dataset from Kaggle to build a predictive binary classification model for a cervical cancer diagnosis using 27 attributes
that potentially have a causal relationship to developing cervical cancer. This project seeks to understand which attributes are most
telling of whether or not a woman will develop cervical cancer through singular value decomposition, support vector machines and artificial
neural networks.

This project was a final project for MATH123: Mathematical Aspects of Data Analysis A, a course at Tufts University


## Files

### CervicalCancer.py
````
Python

Main data analysis: Includes analysis via Support Vector Machine and Artificial Neural Networks (ANN) to 
build a binary classification model to determine whether or not someone would be diagnosed with cervical 
cancer.

The file also includes the data cleaning and reduction that was required due to incomplete/irrelevant aspects of 
the dataset
````

### CervicalCancerSVD.m
````
MATLAB

This portion of the code was used to determine whether or not the data could be compressed through use of 
Singular Value Decomposition.

The code determines the directions of greatest variance in order to make new features; the number of 'relevant'
features is shown via the dropoff graph produced.
````

### ImportantAttributesCervicalCancer.py
````
Python

After performing singular value decomposition on the data, this code was used to determine which attributes 
where most likely to determine a cervical cancer diagnosis through use of a Random Forest Classifier and 
graph that ranks the most relevant features.
````
