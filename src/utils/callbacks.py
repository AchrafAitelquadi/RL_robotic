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
                print(f"📊 ÉVALUATION au step {self.n_calls} ({self.n_eval_episodes} épisodes)")
                print(f"{'='*80}")
            
            for episode_idx in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                step_count = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                    done = terminated or truncated
                
                # Récupérer les infos finales
                is_success = info.get('is_success', False)
                final_distance = info.get('distance_to_target', 0.0)
                
                episode_rewards.append(episode_reward)
                episode_successes.append(1.0 if is_success else 0.0)
                episode_distances.append(final_distance)
                
                # Afficher résultat de cet épisode
                if self.verbose > 0:
                    status_icon = "✅" if is_success else "❌"
                    status_text = "SUCCÈS" if is_success else "ÉCHEC"
                    print(f"  Episode {episode_idx+1:2d}/{self.n_eval_episodes}: {status_icon} {status_text:6s} | "
                          f"Reward: {episode_reward:6.2f} | "
                          f"Distance finale: {final_distance:.4f}m | "
                          f"Steps: {step_count}")
            
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
                print(f"📈 STATISTIQUES GLOBALES:")
                print(f"  Succès: {num_successes}/{self.n_eval_episodes} ({mean_success*100:.1f}%)")
                print(f"  Reward moyen: {mean_reward:.2f} (±{std_reward:.2f})")
                print(f"  Distance moyenne: {mean_distance:.4f}m (±{std_distance:.4f}m)")
                
                if mean_success > 0:
                    print(f"  🎯 Le robot a réussi à pousser l'objet {num_successes} fois !")
                else:
                    print(f"  ⚠️  Le robot n'a pas encore réussi à pousser l'objet vers la cible")
            
            # Sauvegarder le meilleur modèle
            if mean_reward > self.best_mean_reward and self.save_path is not None:
                improvement = mean_reward - self.best_mean_reward
                self.best_mean_reward = mean_reward
                save_path = f"{self.save_path}/best_model"
                self.model.save(save_path)
                if self.verbose > 0:
                    print(f"  ⭐ NOUVEAU RECORD ! Amélioration: +{improvement:.2f}")
                    print(f"  💾 Meilleur modèle sauvegardé: {save_path}")
            
            if self.verbose > 0:
                print(f"{'='*80}\n")
        
        return True
