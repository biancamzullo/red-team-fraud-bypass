import pandas as pd
import numpy as np
import gc 


print("Loading data...")

train_transaction = pd.read_csv('data/train_transaction.csv')
train_identity = pd.read_csv('data/train_identity.csv')


print("Merging data...")

train_df = train_transaction.merge(train_identity, on='TransactionID', how='left')


del train_transaction, train_identity
gc.collect()

print(f"Merged dataset shape: {train_df.shape}")


print("Dropping columns with too many missing values...")
missing_threshold = 0.80
missing_fractions = train_df.isnull().mean()
cols_to_drop = missing_fractions[missing_fractions > missing_threshold].index
train_df = train_df.drop(columns=cols_to_drop)



y = train_df['isFraud']
X = train_df.drop(columns=['isFraud', 'TransactionID']) # TransactionID is an index, not a predictive feature


categorical_cols = X.select_dtypes(include=['object', 'string']).columns

print("Encoding categorical variables...")

for col in categorical_cols:
    X[col] = X[col].astype(str) 
    X[col] = pd.factorize(X[col])[0] 

print(f"Final feature matrix shape: {X.shape}")