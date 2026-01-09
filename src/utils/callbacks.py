"""
Callback personnalisé pour l'évaluation pendant l'entraînement
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EvaluationCallback(BaseCallback):
    """
    Callback pour évaluer l'agent périodiquement et sauvegarder les meilleurs modèles
    """
    
    def __init__(
        self,
        eval_env,
        eval_freq=5000,
        n_eval_episodes=10,
        save_path=None,
        verbose=1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
        self.rewards_history = []
        self.success_history = []
        self.distance_history = []
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Évaluer l'agent
            episode_rewards = []
            episode_successes = []
            episode_distances = []
            
            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"EVALUATION au step {self.n_calls} ({self.n_eval_episodes} episodes)")
                print(f"{'='*80}")
            
            for episode_idx in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                step_count = 0
                min_distance = float('inf')  # Distance minimale atteinte
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                    done = terminated or truncated
                    
                    # Suivre la distance minimale
                    if 'observation' in obs and 'desired_goal' in obs and 'achieved_goal' in obs:
                        current_distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
                        min_distance = min(min_distance, current_distance)
                
                # Recuperer les infos finales
                is_success = info.get('is_success', False) or info.get('is_success', 0.0)
                
                # Calculer distance finale
                if 'observation' in obs and 'desired_goal' in obs and 'achieved_goal' in obs:
                    final_distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
                else:
                    final_distance = info.get('distance_to_target', 0.0)
                
                episode_rewards.append(episode_reward)
                episode_successes.append(1.0 if is_success else 0.0)
                episode_distances.append(final_distance)
                
                # Afficher resultat de cet episode
                if self.verbose > 0:
                    if is_success:
                        status_icon = "[OK] REACH"
                        status_text = "SUCCES"
                    else:
                        status_icon = "[XX] ECHEC"
                        status_text = "ECHEC "
                    
                    print(f"  Ep {episode_idx+1:2d}/{self.n_eval_episodes}: {status_icon} | "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Dist finale: {final_distance:.4f}m | "
                          f"Dist min: {min_distance:.4f}m | "
                          f"Steps: {step_count:2d}")
            
            mean_reward = np.mean(episode_rewards)
            mean_success = np.mean(episode_successes)
            mean_distance = np.mean(episode_distances)
            std_reward = np.std(episode_rewards)
            std_distance = np.std(episode_distances)
            num_successes = int(sum(episode_successes))
            
            self.rewards_history.append(mean_reward)
            self.success_history.append(mean_success)
            self.distance_history.append(mean_distance)
            
            if self.verbose > 0:
                print(f"\n{'-'*80}")
                print(f"STATISTIQUES GLOBALES:")
                print(f"  Taux de succes: {num_successes}/{self.n_eval_episodes} episodes ({mean_success*100:.1f}%)")
                print(f"  Reward moyen: {mean_reward:.2f} (std: {std_reward:.2f})")
                print(f"  Distance finale moyenne: {mean_distance:.4f}m (std: {std_distance:.4f}m)")
                print(f"  Seuil de succes: < 0.05m (5cm)")
                
                if mean_success >= 0.80:
                    print(f"  [EXCELLENT] Le robot atteint la cible {num_successes}/{self.n_eval_episodes} fois!")
                elif mean_success >= 0.50:
                    print(f"  [BON] Le robot progresse bien ({mean_success*100:.0f}% de succes)")
                elif mean_success > 0:
                    print(f"  [EN COURS] Le robot commence a apprendre ({num_successes} succes)")
                else:
                    print(f"  [DEBUT] Le robot explore encore (distance moy: {mean_distance:.4f}m)")
            
            # Sauvegarder le meilleur modèle
            if mean_reward > self.best_mean_reward and self.save_path is not None:
                improvement = mean_reward - self.best_mean_reward
                self.best_mean_reward = mean_reward
                save_path = f"{self.save_path}/best_model"
                self.model.save(save_path)
                if self.verbose > 0:
                    print(f"  [RECORD] Nouveau meilleur score! Amelioration: +{improvement:.2f}")
                    print(f"  [SAVE] Modele sauvegarde: {save_path}")
            
            if self.verbose > 0:
                print(f"{'='*80}\n")
        
        return True
