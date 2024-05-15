# Importing the necessary libraries
import pandas as pd 
import numpy as np 
import sklearn.impute as impute

# Load the dataset
dataset = pd.read_csv('machine-learning-AZ\data-preprocessing\coding-exercises\exercise-2\pima-indians-diabetes.csv')

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isna().sum()
print(missing_data)

# Print the number of missing entries in each column
for column in dataset.columns:
    missing_count = dataset[column].isna().sum()
    print(f"Column '{column}' has {missing_count} missing entries.")

# Configure an instance of the SimpleImputer class
imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(dataset)

# Apply the transform to the DataFrame
dataset = imputer.transform(dataset)

# Print your updated matrix of features
print(dataset)