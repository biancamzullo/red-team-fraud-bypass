import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from data_processing import load_and_clean_data

def get_ensemble_prediction_prob(xgb_model, lgb_model, data_df):
    xgb_probs = xgb_model.predict_proba(data_df)[:, 1]
    lgb_probs = lgb_model.predict_proba(data_df)[:, 1]
    return (xgb_probs + lgb_probs) / 2

def run_adversarial_retraining():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    
  
    transaction_path = os.path.join(current_dir, '..', 'data', 'train_transaction.csv')
    identity_path = os.path.join(current_dir, '..', 'data', 'train_identity.csv')
    X_train, X_test, y_train, y_test = load_and_clean_data(transaction_path, identity_path)
    
    
    adv_attack_path = os.path.join(models_dir, 'successful_adversarial_attacks.csv')
    if not os.path.exists(adv_attack_path):
        print("[-] Error: No successful adversarial attacks found. Run red_team_agent.py first.")
        return
        
    adversarial_data = pd.read_csv(adv_attack_path)
    
    if 'isFraud' in adversarial_data.columns:
        X_adv = adversarial_data.drop(columns=['isFraud'])
        y_adv = adversarial_data['isFraud']
    else:
        X_adv = adversarial_data
       
        y_adv = pd.Series([1] * len(X_adv), name='isFraud')
    
    print(f"\n[+] Ingesting {len(X_adv)} successful bypass transactions into training dataset...")
    
  
    X_train_hardened = pd.concat([X_train, X_adv], ignore_index=True)
    y_train_hardened = pd.concat([y_train, y_adv], ignore_index=True)
    
    imbalance_ratio = (y_train_hardened == 0).sum() / (y_train_hardened == 1).sum()

    print("\n--- Retraining Hardened XGBoost Component ---")
    xgb_hardened = xgb.XGBClassifier(
        n_estimators=200, scale_pos_weight=imbalance_ratio,
        random_state=42, tree_method='hist', n_jobs=-1
    )
    xgb_hardened.fit(X_train_hardened, y_train_hardened)

    print("\n--- Retraining Hardened LightGBM Component ---")
    lgb_hardened = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=8,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio * 0.6, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_hardened.fit(X_train_hardened, y_train_hardened)

    print("\n--- Testing Hardened Model Against the Red Team Vulnerabilities ---")
   
    hardened_probs = get_ensemble_prediction_prob(xgb_hardened, lgb_hardened, X_adv)
    
    print("\n" + "="*50)
    print("VULNERABILITY MITIGATION REPORT")
    print("="*50)
    caught_count = 0
    for i, prob in enumerate(hardened_probs):
        status = "CAUGHT" if prob >= 0.5 else "SLIPPED AGAIN"
        if prob >= 0.5: caught_count += 1
        print(f"Adversarial Sample {i+1}: New Fraud Probability = {prob:.4f} -> [{status}]")
        
    print("-" * 50)
    print(f"Mitigation Success Rate: {(caught_count / len(X_adv)) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_adversarial_retraining()