import os
import joblib  
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from data_processing import load_and_clean_data

def train_and_save_ensemble():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    transaction_path = os.path.join(current_dir, '..', 'data', 'train_transaction.csv')
    identity_path = os.path.join(current_dir, '..', 'data', 'train_identity.csv')
    
    X_train, X_test, y_train, y_test = load_and_clean_data(transaction_path, identity_path)
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    print("\n--- Training Best XGBoost Component ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    print("\n--- Training Best Tuned LightGBM Component ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio * 0.6,
        random_state=42,
        n_jobs=-1,
        verbose=-1 
    )
    lgb_model.fit(X_train, y_train)

    print("\n--- Evaluating Blended Ensemble (Soft Voting) ---")
    
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    
   
    ensemble_probs = (xgb_probs + lgb_probs) / 2
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    print("\n" + "="*50)
    print("FINAL DEFENSE ENSEMBLE RESULTS")
    print("="*50)
    print(f"Ensemble ROC-AUC: {roc_auc_score(y_test, ensemble_probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_preds))
    print("="*50)

   
    models_dir = os.path.join(current_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    
    print("Saving defense components to disk...")
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_component.pkl'))
    joblib.dump(lgb_model, os.path.join(models_dir, 'lgb_component.pkl'))
    
    
    test_df = X_test.copy()
    test_df['isFraud'] = y_test
    fraud_samples = test_df[test_df['isFraud'] == 1].drop(columns=['isFraud']).head(100)
    fraud_samples.to_csv(os.path.join(models_dir, 'fraud_attack_targets.csv'), index=False)
    print("Saved target fraud examples for adversarial testing.")

if __name__ == "__main__":
    train_and_save_ensemble()