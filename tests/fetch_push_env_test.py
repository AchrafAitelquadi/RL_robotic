"""
Test de l'environnement FetchPush personnalisé
Montre comment personnaliser observations, actions, et reward
"""

import sys
import os

# Ajouter le répertoire parent au PYTHONPATH pour pouvoir importer src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import make_fetch_push_env
import numpy as np

print("=" * 80)
print("   TEST ENVIRONNEMENT FETCHPUSH PERSONNALISÉ")
print("=" * 80)

# Test 1: Environnement avec reward SPARSE
print("\n--- Test 1: Reward Sparse ---")
env_sparse = make_fetch_push_env(reward_type='sparse', max_episode_steps=200)
obs, _ = env_sparse.reset()

print(f"Observation keys: {obs.keys()}")
print(f"  observation shape: {obs['observation'].shape}")
print(f"  achieved_goal: {obs['achieved_goal']}")
print(f"  desired_goal: {obs['desired_goal']}")

# Tester quelques actions
for i in range(5):
    action = env_sparse.action_space.sample()
    obs, reward, terminated, truncated, info = env_sparse.step(action)
    print(f"  Step {i}: reward={reward:.3f}, distance={info['distance_to_target']:.4f}m")

env_sparse.close()

# Test 2: Environnement avec reward DENSE
print("\n--- Test 2: Reward Dense ---")
env_dense = make_fetch_push_env(reward_type='dense', max_episode_steps=200)
obs, _ = env_dense.reset()

for i in range(5):
    action = env_dense.action_space.sample()
    obs, reward, terminated, truncated, info = env_dense.step(action)
    print(f"  Step {i}: reward={reward:.3f}, distance={info['distance_to_target']:.4f}m")

env_dense.close()

# Test 3: Environnement avec reward SHAPED (personnalisé)
print("\n--- Test 3: Reward Shaped (Personnalisé) ---")
env_shaped = make_fetch_push_env(reward_type='shaped', max_episode_steps=200)
obs, _ = env_shaped.reset()

print(f"Reward shaped inclut:")
print(f"  - Terme de distance")
print(f"  - Bonus de proximité")
print(f"  - Bonus de succès")
print(f"  - Pénalité temporelle")
print(f"  - Récompense pour approcher l'objet")

for i in range(5):
    action = env_shaped.action_space.sample()
    obs, reward, terminated, truncated, info = env_shaped.step(action)
    print(f"  Step {i}: reward={reward:.3f}, distance_target={info['distance_to_target']:.4f}m, "
          f"distance_object={info['distance_to_object']:.4f}m")

env_shaped.close()

# Test 4: Visualisation
print("\n--- Test 4: Visualisation ---")
print("Création de l'environnement avec visualisation...")
input("Appuyez sur Entrée pour ouvrir la fenêtre MuJoCo...")

env_visual = make_fetch_push_env(
    reward_type='shaped',
    max_episode_steps=100,
    render_mode='human'
)

print("\n✓ Environnement créé avec visualisation")
print("  Exécution de 2 épisodes avec actions aléatoires...")

try:
    for episode in range(2):
        print(f"\n--- Episode {episode + 1} ---")
        obs, _ = env_visual.reset()
        episode_reward = 0
        
        for step in range(100):
            action = env_visual.action_space.sample()
            obs, reward, terminated, truncated, info = env_visual.step(action)
            episode_reward += reward
            
            if step % 20 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={episode_reward:.2f}, "
                      f"distance={info['distance_to_target']:.4f}m")
            
            if terminated or truncated:
                print(f"  Episode terminé au step {step}")
                if info.get('is_success', False):
                    print(f"  ★ SUCCÈS!")
                break
        
        print(f"  Reward total: {episode_reward:.2f}")

except KeyboardInterrupt:
    print("\n✓ Arrêt par l'utilisateur")

env_visual.close()

print("\n" + "=" * 80)
print("✓ TOUS LES TESTS TERMINÉS")
print("=" * 80)

print("\n📝 COMMENT PERSONNALISER:")
print("  1. Ouvrez: src/environment/custom_fetch_push.py")
print("  2. Modifiez:")
print("     - _customize_observation() : Changer les observations")
print("     - _customize_action()      : Modifier les actions")
print("     - _customize_reward()      : Personnaliser le reward")
print("     - _sample_goal()           : Changer la génération de cibles")
