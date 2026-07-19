import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from data_processing import load_and_clean_data
import time
import os

def train_and_evaluate():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
   
    transaction_path = os.path.join(current_dir, '..', 'data', 'train_transaction.csv')
    identity_path = os.path.join(current_dir, '..', 'data', 'train_identity.csv')
    
    
    X_train, X_test, y_train, y_test = load_and_clean_data(
        transaction_path=transaction_path, 
        identity_path=identity_path
    )
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    print("\n--- Training XGBoost ---")
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        scale_pos_weight=imbalance_ratio, 
        random_state=42,
        tree_method='hist', 
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_probs)

    print("\n--- Training LightGBM ---")
    start_time = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_time = time.time() - start_time
    
    lgb_preds = lgb_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_probs)

  
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    
    print(f"XGBoost Training Time: {xgb_time:.2f} seconds")
    print(f"XGBoost ROC-AUC:       {xgb_auc:.4f}")
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, xgb_preds))
    
    print("-" * 50)
    
    print(f"LightGBM Training Time: {lgb_time:.2f} seconds")
    print(f"LightGBM ROC-AUC:       {lgb_auc:.4f}")
    print("\nLightGBM Classification Report:")
    print(classification_report(y_test, lgb_preds))
    print("="*50)

    return xgb_model, lgb_model

if __name__ == "__main__":
    train_and_evaluate()