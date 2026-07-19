import os
import joblib
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')

def get_ensemble_prediction(xgb_model, lgb_model, data_row):
    xgb_prob = xgb_model.predict_proba(data_row)[:, 1][0]
    lgb_prob = lgb_model.predict_proba(data_row)[:, 1][0]
    return (xgb_prob + lgb_prob) / 2

def boundary_attack_agent():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    
    print("Loading Black-Box Defense Ensemble...")
    xgb_model = joblib.load(os.path.join(models_dir, 'xgb_component.pkl'))
    lgb_model = joblib.load(os.path.join(models_dir, 'lgb_component.pkl'))
    
    print("Loading target fraud transactions...")
    targets = pd.read_csv(os.path.join(models_dir, 'fraud_attack_targets.csv'))
    
   
    features_to_attack = ['TransactionAmt', 'C1', 'C2', 'C13']
    
    max_attempts = 100
    successful_bypasses = 0
    bypassed_data = []

    print("\n--- Initiating Red Team Attack Simulation ---")
    
    for index, row in targets.iterrows():
        original_row = pd.DataFrame([row])
        current_prob = get_ensemble_prediction(xgb_model, lgb_model, original_row)
        
        if current_prob < 0.5:
            continue
            
        print(f"\nTarget {index + 1}: Initial Fraud Probability = {current_prob:.4f}")
        
        attack_row = original_row.copy()
        bypassed = False
        
        for attempt in range(max_attempts):
            feat = np.random.choice(features_to_attack)
            
            # WIDEN THE MUTATION: Allow up to a 15% change instead of 5%
            noise_multiplier = np.random.uniform(0.85, 1.15)
            attack_row[feat] = attack_row[feat] * noise_multiplier
            
            new_prob = get_ensemble_prediction(xgb_model, lgb_model, attack_row)
            
            if new_prob < 0.5:
                print(f"  -> SUCCESS on attempt {attempt + 1}! Probability dropped to {new_prob:.4f}")
                successful_bypasses += 1
                
                
                bypassed_data.append(attack_row)
                bypassed = True
                break
                
        if not bypassed:
            print(f"  -> FAILED: Model held strong after {max_attempts} adversarial mutations.")


    if bypassed_data:
        adversarial_df = pd.concat(bypassed_data, ignore_index=True)
        adversarial_df.to_csv(os.path.join(models_dir, 'successful_adversarial_attacks.csv'), index=False)
        print(f"\n[+] Saved {len(adversarial_df)} successful bypasses to models/successful_adversarial_attacks.csv")

    print("\n" + "="*50)
    print("RED TEAM ATTACK REPORT")
    print("="*50)
    print(f"Total Targets Attacked: {len(targets)}")
    print(f"Successful Bypasses:    {successful_bypasses}")
    print(f"Defense Success Rate:   {((len(targets) - successful_bypasses) / len(targets)) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    boundary_attack_agent()