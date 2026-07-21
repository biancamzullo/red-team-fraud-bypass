import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def generate_decision_boundary_plot():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    assets_dir = os.path.join(current_dir, '..', 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    print("Generating Decision Boundary Visualization...")


    xgb_model = joblib.load(os.path.join(models_dir, 'xgb_component.pkl'))
    lgb_model = joblib.load(os.path.join(models_dir, 'lgb_component.pkl'))
    targets = pd.read_csv(os.path.join(models_dir, 'fraud_attack_targets.csv'))
    
    past_attack_file = os.path.join(models_dir, 'successful_adversarial_attacks.csv')
    if os.path.exists(past_attack_file):
        attacks = pd.read_csv(past_attack_file)
    else:
        attacks = targets.copy()

    features_to_attack = ['TransactionAmt', 'C1', 'C2', 'C13']

    
    base_row = targets.iloc[[0]].copy()
    attack_row = attacks.iloc[[-1]].copy()

    
    pca = PCA(n_components=2)
    features_data = pd.concat([targets[features_to_attack], attacks[features_to_attack]], ignore_index=True)
    pca.fit(features_data)

    base_2d = pca.transform(base_row[features_to_attack])
    attack_2d = pca.transform(attack_row[features_to_attack])

    x_min, x_max = min(base_2d[0,0], attack_2d[0,0]) - 50, max(base_2d[0,0], attack_2d[0,0]) + 50
    y_min, y_max = min(base_2d[0,1], attack_2d[0,1]) - 50, max(base_2d[0,1], attack_2d[0,1]) + 50
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

  
    reconstructed_features = pca.inverse_transform(grid_2d)
    reconstructed_df = pd.DataFrame(reconstructed_features, columns=features_to_attack)


    full_grid_df = pd.concat([base_row] * len(reconstructed_df), ignore_index=True)
    for feat in features_to_attack:
        full_grid_df[feat] = reconstructed_df[feat]

  
    xgb_probs = xgb_model.predict_proba(full_grid_df)[:, 1]
    lgb_probs = lgb_model.predict_proba(full_grid_df)[:, 1]
    probs = ((xgb_probs + lgb_probs) / 2).reshape(xx.shape)


    base_prob = ((xgb_model.predict_proba(base_row)[:, 1] + lgb_model.predict_proba(base_row)[:, 1]) / 2)[0]
    attack_prob = ((xgb_model.predict_proba(attack_row)[:, 1] + lgb_model.predict_proba(attack_row)[:, 1]) / 2)[0]


    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)


    contour = ax.contourf(xx, yy, probs, levels=20, cmap='RdYlGn_r', alpha=0.85)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Ensemble Fraud Risk Score', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')


    ax.contour(xx, yy, probs, levels=[0.5], colors='cyan', linewidths=2, linestyles='--')


    ax.scatter(base_2d[0,0], base_2d[0,1], color='red', s=180, edgecolors='white', label=f'Baseline Target (Prob: {base_prob:.2f})', zorder=5)
    ax.scatter(attack_2d[0,0], attack_2d[0,1], color='lime', s=180, marker='*', edgecolors='black', label=f'Evolved Apex Strain (Prob: {attack_prob:.2f})', zorder=5)

    ax.annotate('', xy=(attack_2d[0,0], attack_2d[0,1]), xytext=(base_2d[0,0], base_2d[0,1]),
                arrowprops=dict(arrowstyle="->", color='cyan', lw=2.5, mutation_scale=20))

  
    ax.set_title("Adversarial Perturbation & Decision Boundary Penetration", fontsize=14, fontweight='bold', pad=15, color='white')
    ax.set_xlabel("Principal Component 1 (Attacked Features Subspace)", color='white', fontsize=11)
    ax.set_ylabel("Principal Component 2 (Attacked Features Subspace)", color='white', fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='#111111', edgecolor='cyan')

    plt.tight_layout()
    output_path = os.path.join(assets_dir, 'adversarial_boundary_attack.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[+] High-resolution decision boundary asset generated at: {output_path}")

if __name__ == "__main__":
    generate_decision_boundary_plot()