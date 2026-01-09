"""
Test de l'environnement FetchReach avec visualisation MuJoCo
Lance l'environnement et affiche le robot en action
"""

import numpy as np
import time
from src.environment import make_fetch_reach_env


def test_reach_env_visual(reward_type='dense', n_episodes=5, delay=0.05):
    """
    Teste FetchReach avec visualisation graphique
    
    Args:
        reward_type: Type de reward ('sparse', 'dense', 'dense_normalized', 'shaped')
        n_episodes: Nombre d'épisodes à exécuter
        delay: Délai entre chaque step (secondes) pour ralentir la visualisation
    """
    print("="*80)
    print(f"   TEST FETCHREACH - Reward: {reward_type}")
    print("="*80)
    
    # Créer l'environnement avec rendu visuel
    env = make_fetch_reach_env(
        reward_type=reward_type,
        max_episode_steps=50,
        render_mode='human'  # Active la visualisation MuJoCo
    )
    
    print(f"\n✅ Environnement créé avec succès!")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Reward type: {reward_type}")
    
    for episode in range(n_episodes):
        print(f"\n{'='*80}")
        print(f"📍 ÉPISODE {episode + 1}/{n_episodes}")
        print(f"{'='*80}")
        
        obs, info = env.reset()
        
        # Positions initiales
        gripper_pos = obs['observation'][:3]
        goal_pos = obs['desired_goal']
        initial_distance = np.linalg.norm(gripper_pos - goal_pos)
        
        print(f"🤖 Position gripper: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
        print(f"🎯 Position cible:   [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
        print(f"📏 Distance initiale: {initial_distance:.3f}m")
        
        episode_reward = 0
        step = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Action aléatoire pour tester (normalement: agent entraîné)
            action = env.action_space.sample()
            
            # Exécuter l'action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Calculer la distance actuelle
            current_gripper_pos = obs['observation'][:3]
            current_distance = np.linalg.norm(current_gripper_pos - goal_pos)
            
            # Afficher les infos toutes les 10 steps
            if step % 10 == 0:
                print(f"  Step {step:2d} | Distance: {current_distance:.3f}m | Reward: {reward:7.2f} | Total: {episode_reward:7.2f}")
            
            # Pause pour visualisation
            time.sleep(delay)
        
        # Résultats de l'épisode
        final_distance = np.linalg.norm(obs['observation'][:3] - goal_pos)
        success = info.get('is_success', False)
        
        print(f"\n{'─'*80}")
        print(f"📊 RÉSULTATS ÉPISODE {episode + 1}:")
        print(f"   Statut:           {'✅ SUCCÈS' if success else '❌ ÉCHEC'}")
        print(f"   Steps:            {step}")
        print(f"   Reward total:     {episode_reward:.2f}")
        print(f"   Distance finale:  {final_distance:.3f}m (seuil: 0.05m)")
        print(f"   Progression:      {(initial_distance - final_distance):.3f}m")
        print(f"{'─'*80}")
        
        # Pause entre les épisodes
        if episode < n_episodes - 1:
            print("\n⏸️  Appuyez sur ENTER pour l'épisode suivant...")
            input()
    
    env.close()
    print("\n" + "="*80)
    print("✅ Test terminé!")
    print("="*80)


def test_reach_all_rewards(n_episodes=2):
    """
    Teste tous les types de rewards avec visualisation
    """
    reward_types = ['sparse', 'dense', 'dense_normalized', 'shaped']
    
    for reward_type in reward_types:
        print("\n\n")
        test_reach_env_visual(reward_type=reward_type, n_episodes=n_episodes, delay=0.02)
        
        if reward_type != reward_types[-1]:
            print("\n⏸️  Appuyez sur ENTER pour tester le prochain reward...")
            input()


def quick_test():
    """Test rapide sans visualisation pour vérifier que tout fonctionne"""
    print("🔧 Test rapide (sans visualisation)...\n")
    
    env = make_fetch_reach_env(reward_type='dense', max_episode_steps=50)
    
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        success = info.get('is_success', False)
        print(f"  Épisode {episode + 1}: {'✅ Succès' if success else '❌ Échec'} | "
              f"Reward: {episode_reward:.2f} | Steps: {step + 1}")
    
    env.close()
    print("\n✅ Test rapide OK!\n")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("   TEST ENVIRONNEMENT FETCHREACH")
    print("="*80)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            # Tester tous les rewards
            test_reach_all_rewards(n_episodes=2)
        elif sys.argv[1] == '--quick':
            # Test rapide
            quick_test()
        elif sys.argv[1] in ['sparse', 'dense', 'dense_normalized', 'shaped']:
            # Tester un reward spécifique
            reward = sys.argv[1]
            n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            test_reach_env_visual(reward_type=reward, n_episodes=n_ep)
        else:
            print("Usage:")
            print("  python tests/fetch_reach_env_test.py                    # Test par défaut (dense, 5 épisodes)")
            print("  python tests/fetch_reach_env_test.py --all              # Teste tous les rewards")
            print("  python tests/fetch_reach_env_test.py --quick            # Test rapide sans visualisation")
            print("  python tests/fetch_reach_env_test.py <reward> [n_ep]   # Test avec reward spécifique")
            print("\nRewards disponibles: sparse, dense, dense_normalized, shaped")
    else:
        # Test par défaut: dense avec 5 épisodes
        test_reach_env_visual(reward_type='dense', n_episodes=5, delay=0.05)
