import pandas as pd

# Load the Iris dataset into a Pandas DataFrame
df = pd.read_csv('iris.csv', header=None)

# Display the first few rows of the dataset
print(df.head())
