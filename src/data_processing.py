import pandas as pd
import gc
from sklearn.model_selection import train_test_split

def load_and_clean_data(transaction_path='data/train_transaction.csv', identity_path='data/train_identity.csv'):
    print("Loading datasets...")
    train_transaction = pd.read_csv(transaction_path)
    train_identity = pd.read_csv(identity_path)

    print("Merging datasets...")
    df = train_transaction.merge(train_identity, on='TransactionID', how='left')
    
    del train_transaction, train_identity
    gc.collect()

    print("Dropping columns with > 80% missing values...")
    missing_threshold = 0.80
    missing_fractions = df.isnull().mean()
    cols_to_drop = missing_fractions[missing_fractions > missing_threshold].index
    df = df.drop(columns=cols_to_drop)

    y = df['isFraud']
    X = df.drop(columns=['isFraud', 'TransactionID'])

    print("Encoding categorical variables...")
   
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        X[col] = pd.factorize(X[col])[0]

  
    X = X.fillna(-999)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
   
    X_tr, X_te, y_tr, y_te = load_and_clean_data()
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")