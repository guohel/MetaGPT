import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import chardet


# Load the CSV file
file_path = '/home/gee/repos/MetaGPT/test1.csv'

# First, detect the encoding
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())  # or f.read(10000) to read first 10000 bytes
# Now, use the detected encoding to read the CSV
data = pd.read_csv(file_path, encoding=result['encoding'])

# Display the first few rows of the dataset
data.head()

# Check the data types of the columns
data_types = data.dtypes
print(data_types)

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

# Exploratory Data Analysis on numerical columns
# Correlation matrix
correlation_matrix = data[numerical_cols].corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Analyzing categorical columns
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x=col, hue='NG_Product')  # Assuming 'NG_Product' is the target column
    plt.title(f'Distribution of {col} with respect to NG Products')
    plt.xticks(rotation=45)
    plt.show()

# Summary statistics for numerical columns
summary_statistics = data[numerical_cols].describe()
print(summary_statistics)