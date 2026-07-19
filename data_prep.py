import pandas as pd
import numpy as np
import gc # Garbage collector to manage memory

# 1. Load the data
print("Loading data...")
# Update these paths if your extracted files are in a different location inside your 'data/' folder
train_transaction = pd.read_csv('data/train_transaction.csv')
train_identity = pd.read_csv('data/train_identity.csv')

# 2. Merge the datasets
print("Merging data...")
# We use a left join because every transaction needs to be evaluated, even if it lacks identity info
train_df = train_transaction.merge(train_identity, on='TransactionID', how='left')

# 3. Free up RAM immediately (these datasets are large!)
del train_transaction, train_identity
gc.collect()

print(f"Merged dataset shape: {train_df.shape}")

# 1. Drop features with > 80% missing values
print("Dropping columns with too many missing values...")
missing_threshold = 0.80
missing_fractions = train_df.isnull().mean()
cols_to_drop = missing_fractions[missing_fractions > missing_threshold].index
train_df = train_df.drop(columns=cols_to_drop)

# 2. Separate Features (X) and Target (y)
# The target variable predicting if a transaction is fraudulent is 'isFraud'
y = train_df['isFraud']
X = train_df.drop(columns=['isFraud', 'TransactionID']) # TransactionID is an index, not a predictive feature

# 3. Handle Categorical Data
# We need to convert text columns to numbers so our models can read them
categorical_cols = X.select_dtypes(include=['object', 'string']).columns

print("Encoding categorical variables...")
# For a baseline, we'll use a simple label encoder. Later you might want to try One-Hot Encoding for specific columns.
for col in categorical_cols:
    X[col] = X[col].astype(str) # ensure everything is a string first
    # Factorize assigns a unique integer to each category
    X[col] = pd.factorize(X[col])[0] 

print(f"Final feature matrix shape: {X.shape}")