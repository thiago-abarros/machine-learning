# Importing the necessary libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

# Loading the Iris dataset
dataset = pd.read_csv('machine-learning-AZ\data-preprocessing\coding-exercises\iris.csv')

# Creating the matrix of features (X) and the dependent variable vector (y)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Printing the matrix of features and the dependent variable vector
print(X)
print(y)