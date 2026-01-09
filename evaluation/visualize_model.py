"""
Visualisation d'un modèle entraîné
Montre le robot en action avec rendu graphique MuJoCo
"""

import argparse
import numpy as np
import time
from pathlib import Path

from stable_baselines3 import TD3, SAC
from src.environment import make_fetch_push_env


def visualize_trained_model(
    model_path,
    n_episodes=5,
    reward_type='sparse',
    max_episode_steps=50,
    deterministic=True,
    delay=0.02,
    verbose=True
):
    """
    Visualise un modèle entraîné avec rendu graphique.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé (.zip)
        n_episodes: Nombre d'épisodes à visualiser
        reward_type: Type de reward ('sparse', 'dense', etc.)
        max_episode_steps: Nombre max de steps par épisode
        deterministic: Si True, désactive le bruit d'exploration
        delay: Délai entre chaque step (en secondes) pour ralentir l'animation
        verbose: Afficher les détails
    
    Returns:
        dict: Statistiques des épisodes visualisés
    """
    
    if verbose:
        print("=" * 80)
        print(f"   VISUALISATION DU MODÈLE")
        print("=" * 80)
        print(f"\nModèle: {model_path}")
        print(f"Épisodes: {n_episodes}")
        print(f"Reward type: {reward_type}")
        print(f"Max steps: {max_episode_steps}")
        print(f"Deterministic: {deterministic}")
        print(f"Delay: {delay}s par step")
    
    # Charger le modèle
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    # Détecter l'algorithme
    if 'TD3' in str(model_path) or 'td3' in str(model_path):
        model = TD3.load(model_path)
        algo = 'TD3'
    elif 'SAC' in str(model_path) or 'sac' in str(model_path):
        model = SAC.load(model_path)
        algo = 'SAC'
    else:
        try:
            model = SAC.load(model_path)
            algo = 'SAC'
        except:
            model = TD3.load(model_path)
            algo = 'TD3'
    
    if verbose:
        print(f"\n✓ Modèle chargé: {algo}")
    
    # Créer l'environnement avec visualisation
    if verbose:
        print(f"\n🖼️  Création de l'environnement avec rendu graphique...")
        print(f"   Une fenêtre MuJoCo va s'ouvrir...")
    
    env = make_fetch_push_env(
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
        render_mode='human'
    )
    
    if verbose:
        print(f"\n✓ Environnement créé")
        input("\n⏸️  Appuyez sur ENTRÉE pour commencer la visualisation...")
    
    # Stocker les résultats
    episode_rewards = []
    episode_successes = []
    final_distances = []
    
    try:
        for episode in range(n_episodes):
            if verbose:
                print(f"\n{'='*80}")
                print(f"ÉPISODE {episode + 1}/{n_episodes}")
                print(f"{'='*80}")
            
            obs, info = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            # Afficher infos initiales
            if verbose:
                grip_pos = obs['observation'][:3]
                object_pos = obs['observation'][3:6]
                goal_pos = obs['desired_goal']
                print(f"\n📍 POSITIONS INITIALES:")
                print(f"  Gripper:  [{grip_pos[0]:.3f}, {grip_pos[1]:.3f}, {grip_pos[2]:.3f}]")
                print(f"  Objet:    [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]")
                print(f"  Cible:    [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
                print(f"  Distance initiale: {np.linalg.norm(object_pos - goal_pos):.4f}m")
                print(f"\n🎬 DÉROULEMENT:\n")
            
            while not done:
                # Prédire l'action
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Exécuter l'action
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Afficher infos tous les 10 steps
                if verbose and step_count % 10 == 0:
                    dist_to_target = info.get('distance_to_target', 0)
                    dist_to_object = info.get('distance_to_object', 0)
                    print(f"  Step {step_count:3d}: "
                          f"reward={reward:6.2f} | "
                          f"total={episode_reward:7.2f} | "
                          f"dist_cible={dist_to_target:.4f}m | "
                          f"dist_objet={dist_to_object:.4f}m")
                
                # Ralentir l'animation
                if delay > 0:
                    time.sleep(delay)
                
                done = terminated or truncated
            
            # Résultats de l'épisode
            success = info.get('is_success', False)
            final_distance = info.get('distance_to_target', 0)
            
            episode_rewards.append(episode_reward)
            episode_successes.append(1 if success else 0)
            final_distances.append(final_distance)
            
            if verbose:
                print(f"\n{'─'*80}")
                print(f"📊 RÉSULTATS DE L'ÉPISODE {episode + 1}:")
                print(f"  Statut:           {'✅ SUCCÈS' if success else '❌ ÉCHEC'}")
                print(f"  Reward total:     {episode_reward:.2f}")
                print(f"  Steps:            {step_count}")
                print(f"  Distance finale:  {final_distance:.4f}m")
                print(f"{'─'*80}")
                
                if episode < n_episodes - 1:
                    input(f"\n⏸️  Appuyez sur ENTRÉE pour l'épisode suivant (ou Ctrl+C pour arrêter)...")
    
    except KeyboardInterrupt:
        if verbose:
            print(f"\n\n⚠️  Visualisation arrêtée par l'utilisateur")
    
    finally:
        env.close()
    
    # Statistiques finales
    if verbose and len(episode_rewards) > 0:
        print(f"\n{'='*80}")
        print(f"STATISTIQUES FINALES ({len(episode_rewards)} épisodes)")
        print(f"{'='*80}")
        print(f"\n📊 RÉSUMÉ:")
        print(f"  Taux de succès:      {np.mean(episode_successes)*100:.1f}% "
              f"({int(np.sum(episode_successes))}/{len(episode_rewards)})")
        print(f"  Reward moyen:        {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
        print(f"  Distance finale moy: {np.mean(final_distances):.4f}m (±{np.std(final_distances):.4f}m)")
        print(f"{'='*80}\n")
    
    stats = {
        'n_episodes': len(episode_rewards),
        'success_rate': np.mean(episode_successes) * 100 if episode_successes else 0,
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'mean_final_distance': np.mean(final_distances) if final_distances else 0,
    }
    
    return stats


def main():
    """Point d'entrée en ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Visualiser un modèle TD3 ou SAC entraîné sur FetchPush'
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Chemin vers le modèle (.zip)'
    )
    
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=5,
        help='Nombre d\'épisodes à visualiser (défaut: 5)'
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
        '--delay',
        type=float,
        default=0.02,
        help='Délai entre chaque step en secondes (défaut: 0.02)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Mode rapide (delay=0)'
    )
    
    parser.add_argument(
        '--slow',
        action='store_true',
        help='Mode lent (delay=0.1)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Mode silencieux'
    )
    
    args = parser.parse_args()
    
    # Ajuster le delay selon les flags
    if args.fast:
        delay = 0
    elif args.slow:
        delay = 0.1
    else:
        delay = args.delay
    
    # Visualiser le modèle
    stats = visualize_trained_model(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        reward_type=args.reward_type,
        max_episode_steps=args.max_episode_steps,
        deterministic=not args.stochastic,
        delay=delay,
        verbose=not args.quiet
    )
    
    return stats


if __name__ == '__main__':
    main()
