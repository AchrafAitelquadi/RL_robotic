"""
Évaluation quantitative d'un modèle entraîné
Calcule les métriques de performance: success rate, reward moyen, distance finale, etc.
"""

import argparse
import numpy as np
import time
from pathlib import Path

from stable_baselines3 import TD3, SAC
from src.environment import make_fetch_push_env


def evaluate_trained_model(
    model_path,
    n_eval_episodes=100,
    reward_type='sparse',
    max_episode_steps=50,
    deterministic=True,
    verbose=True
):
    """
    Évalue un modèle entraîné sur plusieurs épisodes.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé (.zip)
        n_eval_episodes: Nombre d'épisodes d'évaluation
        reward_type: Type de reward ('sparse', 'dense', etc.)
        max_episode_steps: Nombre max de steps par épisode
        deterministic: Si True, désactive le bruit d'exploration
        verbose: Afficher les détails
    
    Returns:
        dict: Statistiques d'évaluation
    """
    
    if verbose:
        print("=" * 80)
        print(f"   ÉVALUATION DU MODÈLE")
        print("=" * 80)
        print(f"\nModèle: {model_path}")
        print(f"Épisodes: {n_eval_episodes}")
        print(f"Reward type: {reward_type}")
        print(f"Max steps: {max_episode_steps}")
        print(f"Deterministic: {deterministic}")
    
    # Charger le modèle
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    # Détecter l'algorithme (TD3 ou SAC) depuis le nom du fichier
    if 'TD3' in str(model_path) or 'td3' in str(model_path):
        model = TD3.load(model_path)
        algo = 'TD3'
    elif 'SAC' in str(model_path) or 'sac' in str(model_path):
        model = SAC.load(model_path)
        algo = 'SAC'
    else:
        # Essayer SAC par défaut
        try:
            model = SAC.load(model_path)
            algo = 'SAC'
        except:
            model = TD3.load(model_path)
            algo = 'TD3'
    
    if verbose:
        print(f"\n✓ Modèle chargé: {algo}")
    
    # Créer l'environnement d'évaluation
    env = make_fetch_push_env(
        reward_type=reward_type,
        max_episode_steps=max_episode_steps
    )
    
    # Stocker les résultats
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    final_distances = []
    
    # Évaluer sur plusieurs épisodes
    if verbose:
        print(f"\n{'='*80}")
        print(f"DÉBUT DE L'ÉVALUATION ({n_eval_episodes} épisodes)")
        print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Prédire l'action (sans bruit si deterministic=True)
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Exécuter l'action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            done = terminated or truncated
        
        # Enregistrer les résultats
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        episode_successes.append(1 if info.get('is_success', False) else 0)
        final_distances.append(info.get('distance_to_target', 0))
        
        # Affichage progressif
        if verbose and (episode + 1) % 10 == 0:
            current_success_rate = np.mean(episode_successes) * 100
            current_avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode+1}/{n_eval_episodes}: "
                  f"Success rate = {current_success_rate:.1f}% | "
                  f"Avg reward = {current_avg_reward:.2f}")
    
    elapsed_time = time.time() - start_time
    
    env.close()
    
    # Calculer les statistiques
    stats = {
        'n_episodes': n_eval_episodes,
        'success_rate': np.mean(episode_successes) * 100,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_final_distance': np.mean(final_distances),
        'std_final_distance': np.std(final_distances),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'total_successes': int(np.sum(episode_successes)),
        'evaluation_time': elapsed_time,
    }
    
    # Afficher les résultats
    if verbose:
        print(f"\n{'='*80}")
        print(f"RÉSULTATS DE L'ÉVALUATION")
        print(f"{'='*80}")
        print(f"\n📊 STATISTIQUES GLOBALES:")
        print(f"  Taux de succès:      {stats['success_rate']:.2f}% ({stats['total_successes']}/{n_eval_episodes})")
        print(f"  Reward moyen:        {stats['mean_reward']:.2f} (±{stats['std_reward']:.2f})")
        print(f"  Reward min/max:      {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"  Longueur moyenne:    {stats['mean_episode_length']:.1f} steps")
        print(f"  Distance finale moy: {stats['mean_final_distance']:.4f}m (±{stats['std_final_distance']:.4f}m)")
        print(f"  Temps d'évaluation:  {stats['evaluation_time']:.1f}s")
        
        # Évaluation qualitative
        print(f"\n🎯 ÉVALUATION QUALITATIVE:")
        if stats['success_rate'] >= 80:
            print(f"  ★★★★★ EXCELLENT - Le modèle maîtrise la tâche!")
        elif stats['success_rate'] >= 60:
            print(f"  ★★★★☆ TRÈS BON - Performance solide")
        elif stats['success_rate'] >= 40:
            print(f"  ★★★☆☆ BON - Performance acceptable")
        elif stats['success_rate'] >= 20:
            print(f"  ★★☆☆☆ MOYEN - Nécessite plus d'entraînement")
        elif stats['success_rate'] >= 5:
            print(f"  ★☆☆☆☆ FAIBLE - Début d'apprentissage")
        else:
            print(f"  ☆☆☆☆☆ TRÈS FAIBLE - Modèle non entraîné")
        
        print(f"{'='*80}\n")
    
    return stats


def main():
    """Point d'entrée en ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Évaluer un modèle TD3 ou SAC entraîné sur FetchPush'
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Chemin vers le modèle (.zip)'
    )
    
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100,
        help='Nombre d\'épisodes d\'évaluation (défaut: 100)'
    )
    
    parser.add_argument(
        '--reward-type',
        type=str,
        default='sparse',
        choices=['sparse', 'dense', 'dense_normalized', 'two_stage', 'shaped', 'potential_based'],
        help='Type de reward (défaut: sparse)'
    )
    
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=50,
        help='Nombre max de steps par épisode (défaut: 50)'
    )
    
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Activer le bruit d\'exploration (défaut: deterministic)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Mode silencieux (pas de verbose)'
    )
    
    args = parser.parse_args()
    
    # Évaluer le modèle
    stats = evaluate_trained_model(
        model_path=args.model_path,
        n_eval_episodes=args.n_episodes,
        reward_type=args.reward_type,
        max_episode_steps=args.max_episode_steps,
        deterministic=not args.stochastic,
        verbose=not args.quiet
    )
    
    return stats


if __name__ == '__main__':
    main()
