import os
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def get_ensemble_prediction(xgb_model, lgb_model, data_df):
    xgb_probs = xgb_model.predict_proba(data_df)[:, 1]
    lgb_probs = lgb_model.predict_proba(data_df)[:, 1]
    return (xgb_probs + lgb_probs) / 2

def run_genetic_breeding_simulation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    
    print("[💀] Initializing Adaptive Evolutionary Super-Fraud Breeding Protocol...")
    xgb_model = joblib.load(os.path.join(models_dir, 'xgb_component.pkl'))
    lgb_model = joblib.load(os.path.join(models_dir, 'lgb_component.pkl'))
    targets = pd.read_csv(os.path.join(models_dir, 'fraud_attack_targets.csv'))
    
    features_to_attack = ['TransactionAmt', 'C1', 'C2', 'C13']
    
    # Baseline Hyperparameters
    POPULATION_SIZE = 100
    GENERATIONS = 100
    BASE_MUTATION_RATE = 0.3
    BASE_RADIUS = 0.15  # 15% modification boundary
    
    evolved_bypasses = []
    base_fraud = pd.DataFrame([targets.iloc[0]])
    initial_prob = get_ensemble_prediction(xgb_model, lgb_model, base_fraud)[0]
    print(f"[!] Target Acquired. Baseline Ensemble Fraud Probability: {initial_prob:.4f}")

    # Initialize Population
    population = []
    for _ in range(POPULATION_SIZE):
        clone = base_fraud.copy()
        for feat in features_to_attack:
            clone[feat] = clone[feat] * np.random.uniform(1 - BASE_RADIUS, 1 + BASE_RADIUS)
        population.append(clone)

    best_fitness_history = []
    stagnation_counter = 0

    # Evolutionary Loop
    for gen in range(GENERATIONS):
        pop_df = pd.concat(population, ignore_index=True)
        probs = get_ensemble_prediction(xgb_model, lgb_model, pop_df)
        
        # Fitness = inverse of probability (lower prob = higher fitness)
        fitness_scores = 1.0 - probs
        best_idx = np.argmax(fitness_scores)
        current_best_prob = probs[best_idx]
        
        # Track Stagnation & Dynamically Adjust Hyperparameters
        if len(best_fitness_history) > 0 and np.isclose(fitness_scores[best_idx], max(best_fitness_history), atol=1e-5):
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            
        best_fitness_history.append(fitness_scores[best_idx])
        
        # Adaptive Mutation Mechanics (The Chaos Trigger)
        if stagnation_counter >= 5:
            current_mutation_rate = min(BASE_MUTATION_RATE * 2.5, 0.8)
            current_radius = BASE_RADIUS * 2.6  # Expands exploration boundary to ~40%
            status_flag = f"[⚠️ STAGNATION DETECTED - HYPER-MUTATION ACTIVE | Rate: {current_mutation_rate:.2f}, Radius: {current_radius:.2f}]"
        else:
            current_mutation_rate = BASE_MUTATION_RATE
            current_radius = BASE_RADIUS
            status_flag = "[NORMAL SELECTION]"

        print(f"Generation {gen+1:03d} | Fittest Prob: {current_best_prob:.4f} | {status_flag}")
        
        if current_best_prob < 0.5:
            print(f"[☣️] APEX PREDATOR EVOLVED! Fraud bypassed ensemble at Gen {gen+1} with Prob: {current_best_prob:.4f}")
            evolved_bypasses.append(pd.DataFrame([pop_df.iloc[best_idx]]))
            break

        # Selection (Top 25% survivors)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        survivors = [population[i] for i in sorted_indices[:POPULATION_SIZE // 4]]
        
        # Breeding Block
        next_generation = []
        num_survivors = len(survivors)
        
        while len(next_generation) < POPULATION_SIZE:
            idx_a = np.random.randint(0, num_survivors)
            idx_b = np.random.randint(0, num_survivors)
            parent_a = survivors[idx_a]
            parent_b = survivors[idx_b]
            
            # Crossover
            child = parent_a.copy()
            for feat in features_to_attack:
                if np.random.rand() > 0.5:
                    child[feat] = parent_b[feat].values[0]
            
            # Adaptive Mutation application
            if np.random.rand() < current_mutation_rate:
                mutated_feat = np.random.choice(features_to_attack)
                child[mutated_feat] = child[mutated_feat] * np.random.uniform(1 - current_radius, 1 + current_radius)
                
            next_generation.append(child)
            
        population = next_generation

    # Export Loop
    if evolved_bypasses or current_best_prob < 0.70:
        export_target = pd.DataFrame([pop_df.iloc[best_idx]]) if not evolved_bypasses else pd.concat(evolved_bypasses, ignore_index=True)
        past_attack_file = os.path.join(models_dir, 'successful_adversarial_attacks.csv')
        
        if os.path.exists(past_attack_file):
            existing_df = pd.read_csv(past_attack_file)
            final_df = pd.concat([existing_df, export_target], ignore_index=True).drop_duplicates()
        else:
            final_df = export_target
            
        final_df.to_csv(past_attack_file, index=False)
        print(f"[+] Highly optimized adversarial strain stored inside data stream.")
    else:
        print("[-] Evolution timed out without creating a meaningful vulnerability strain.")

if __name__ == "__main__":
    run_genetic_breeding_simulation()