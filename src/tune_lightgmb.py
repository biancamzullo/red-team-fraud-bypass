import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from data_processing import load_and_clean_data
import os

def train_tuned_lightgbm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    transaction_path = os.path.join(current_dir, '..', 'data', 'train_transaction.csv')
    identity_path = os.path.join(current_dir, '..', 'data', 'train_identity.csv')
    
    X_train, X_test, y_train, y_test = load_and_clean_data(transaction_path, identity_path)
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    print("\n--- Training Tuned LightGBM ---")
  
    tuned_lgb = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,        # Slower learning rate for better generalization
        num_leaves=63,             # Limits the complexity of the trees (default is 31, but let's control depth)
        max_depth=8,               # Explicitly cap the depth to stop overfitting
        min_child_samples=50,      # Ensure leaves have enough data points
        subsample=0.8,             # Row subsampling
        colsample_bytree=0.8,      # Feature subsampling
        scale_pos_weight=imbalance_ratio * 0.6, # Tone down the weight slightly to drastically reduce false positives
        reg_alpha=0.1,             # L1 regularization
        reg_lambda=1.0,            # L2 regularization
        random_state=42,
        n_jobs=-1
    )
    
    tuned_lgb.fit(X_train, y_train)
    
    probs = tuned_lgb.predict_proba(X_test)[:, 1]
  
    preds = (probs >= 0.5).astype(int) 
    
    print("\n" + "="*50)
    print("TUNED LIGHTGBM RESULTS")
    print("="*50)
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("="*50)
    
    return tuned_lgb

if __name__ == "__main__":
    train_tuned_lightgbm()