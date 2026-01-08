"""
Utilitaires pour logging et sauvegarde des résultats
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt


def create_experiment_dir(algorithm, env_name):
    """Créer un dossier pour l'expérience"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{algorithm}_{env_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    return exp_dir


def save_config(exp_dir, config):
    """Sauvegarder la configuration de l'expérience"""
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration sauvegardée: {config_path}")


def save_results(exp_dir, results):
    """Sauvegarder les résultats d'entraînement"""
    results_path = os.path.join(exp_dir, "results.json")
    
    # Convertir numpy arrays en listes
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f"Résultats sauvegardés: {results_path}")


def plot_training_curves(exp_dir, rewards, success_rates, distances):
    """Tracer les courbes d'apprentissage"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].plot(rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True)
    
    # Success Rate
    axes[1].plot(success_rates)
    axes[1].set_title('Success Rate')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success Rate')
    axes[1].grid(True)
    
    # Distance
    axes[2].plot(distances)
    axes[2].set_title('Final Distance to Goal')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Distance (m)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "plots", "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Courbes sauvegardées: {plot_path}")
