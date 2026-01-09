"""
Script principal pour lancer les expériences d'entraînement
Utilise Stable-Baselines3 (SB3) pour TD3 et SAC
"""

import argparse
import sys
import os
import numpy as np
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from src.environment import make_fetch_push_env, make_fetch_reach_env
from src.utils import (
    create_experiment_dir,
    save_config,
    save_results,
    plot_training_curves,
    EvaluationCallback
)


def parse_args():
    """Parser les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Entraîner TD3 ou SAC sur FetchPush')
    
    # Algorithme
    parser.add_argument(
        '--algo',
        type=str,
        required=True,
        choices=['TD3', 'SAC'],
        help='Algorithme à utiliser (TD3 ou SAC)'
    )
    
    # Environnement
    parser.add_argument(
        '--env',
        type=str,
        default='FetchPush',
        choices=['FetchPush', 'FetchReach'],
        help='Nom de l\'environnement (FetchPush ou FetchReach)'
    )
    
    parser.add_argument(
        '--reward-type',
        type=str,
        default='sparse',
        choices=['sparse', 'dense', 'dense_normalized', 'two_stage', 'shaped', 'potential_based'],
        help='Type de reward'
    )
    
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=200,
        help='Nombre maximum de steps par épisode'
    )
    
    # Entraînement
    parser.add_argument(
        '--timesteps',
        type=int,
        default=500000,
        help='Nombre total de timesteps d\'entraînement'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size'
    )
    
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=1000000,
        help='Taille du replay buffer'
    )
    
    parser.add_argument(
        '--learning-starts',
        type=int,
        default=1000,
        help='Nombre de steps avant de commencer l\'apprentissage (defaut: 1000)'
    )
    
    parser.add_argument(
        '--use-her',
        action='store_true',
        help='Utiliser HER (Hindsight Experience Replay)'
    )
    
    parser.add_argument(
        '--her-goal-selection-strategy',
        type=str,
        default='future',
        choices=['future', 'final', 'episode'],
        help='Stratégie de sélection de goal pour HER'
    )
    
    parser.add_argument(
        '--n-sampled-goal',
        type=int,
        default=4,
        help='Nombre de goals virtuels par transition (HER)'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='Discount factor'
    )
    
    parser.add_argument(
        '--tau',
        type=float,
        default=0.05,
        help='Soft update coefficient'
    )
    
    # TD3 spécifique
    parser.add_argument(
        '--policy-delay',
        type=int,
        default=2,
        help='TD3: Fréquence de mise à jour de la policy'
    )
    
    parser.add_argument(
        '--target-policy-noise',
        type=float,
        default=0.2,
        help='TD3: Bruit pour target policy smoothing'
    )
    
    # SAC spécifique
    parser.add_argument(
        '--ent-coef',
        type=str,
        default='auto',
        help='SAC: Coefficient d\'entropie (auto ou float)'
    )
    
    # Architecture des réseaux de neurones
    parser.add_argument(
        '--net-arch',
        type=str,
        default='standard',
        choices=['standard', 'compact', 'large', 'deep', 'wide'],
        help='Architecture des réseaux de neurones'
    )
    
    parser.add_argument(
        '--activation-fn',
        type=str,
        default='relu',
        choices=['relu', 'tanh', 'elu', 'leaky_relu'],
        help='Fonction d\'activation des couches cachées'
    )
    
    # Évaluation
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=5000,
        help='Fréquence d\'évaluation (en timesteps)'
    )
    
    parser.add_argument(
        '--n-eval-episodes',
        type=int,
        default=10,
        help='Nombre d\'épisodes pour l\'évaluation'
    )
    
    # Seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Autres
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0, 1, ou 2)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device pour PyTorch'
    )
    
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Activer TensorBoard pour visualisation'
    )
    
    return parser.parse_args()


def get_policy_kwargs(args):
    """
    Créer le dictionnaire policy_kwargs pour configurer les réseaux de neurones.
    
    Configurations disponibles:
    - standard: [256, 256] - Configuration par défaut SB3
    - compact: [128, 128] - Plus léger, apprentissage plus rapide
    - large: [400, 300] - Style DDPG original, plus de capacité
    - deep: [256, 256, 128] - 3 couches, plus de profondeur
    - wide: [512, 512] - Couches larges, haute capacité
    """
    
    # Dictionnaire des architectures prédéfinies
    ARCHITECTURES = {
        'standard': dict(
            net_arch=[256, 256],
            description="Standard SB3 (2×256)"
        ),
        'compact': dict(
            net_arch=[128, 128],
            description="Compact (2×128) - rapide"
        ),
        'large': dict(
            net_arch=[400, 300],
            description="Large (400+300) - DDPG style"
        ),
        'deep': dict(
            net_arch=[256, 256, 128],
            description="Deep (3 layers) - profondeur"
        ),
        'wide': dict(
            net_arch=[512, 512],
            description="Wide (2×512) - haute capacité"
        ),
    }
    
    # Fonction d'activation
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU,
    }
    
    # Récupérer l'architecture choisie
    arch_config = ARCHITECTURES[args.net_arch]
    
    # Créer le policy_kwargs
    policy_kwargs = dict(
        net_arch=arch_config['net_arch'],
        activation_fn=ACTIVATIONS[args.activation_fn],
    )
    
    return policy_kwargs, arch_config['description']


def create_env(args):
    """Créer l'environnement d'entraînement"""
    if args.env == 'FetchPush':
        env = make_fetch_push_env(
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps
        )
    elif args.env == 'FetchReach':
        env = make_fetch_reach_env(
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps
        )
    else:
        raise ValueError(f"Environnement non supporté: {args.env}")
    return env


def create_eval_env(args):
    """Créer l'environnement d'évaluation"""
    if args.env == 'FetchPush':
        eval_env = make_fetch_push_env(
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps
        )
    elif args.env == 'FetchReach':
        eval_env = make_fetch_reach_env(
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps
        )
    else:
        raise ValueError(f"Environnement non supporté: {args.env}")
    return eval_env


def create_model(args, env, tensorboard_log_path=None):
    """Créer le modèle SB3"""
    
    # Récupérer la configuration des réseaux de neurones
    policy_kwargs, arch_description = get_policy_kwargs(args)
    
    if args.algo == 'TD3':
        # Bruit d'action pour exploration
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        # Configuration HER si activé
        if args.use_her:
            replay_buffer_class = HerReplayBuffer
            replay_buffer_kwargs = dict(
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy=args.her_goal_selection_strategy,
            )
        else:
            replay_buffer_class = None
            replay_buffer_kwargs = None
        
        model = TD3(
            policy='MultiInputPolicy',
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            policy_delay=args.policy_delay,
            target_policy_noise=args.target_policy_noise,
            target_noise_clip=0.5,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log_path,
            verbose=args.verbose,
            seed=args.seed,
            device=args.device
        )
        
    elif args.algo == 'SAC':
        # Convertir ent_coef
        if args.ent_coef == 'auto':
            ent_coef = 'auto'
        else:
            ent_coef = float(args.ent_coef)
        
        # Configuration HER si activé
        if args.use_her:
            replay_buffer_class = HerReplayBuffer
            replay_buffer_kwargs = dict(
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy=args.her_goal_selection_strategy,
            )
        else:
            replay_buffer_class = None
            replay_buffer_kwargs = None
        
        model = SAC(
            policy='MultiInputPolicy',
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            ent_coef=ent_coef,
            target_update_interval=1,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log_path,
            verbose=args.verbose,
            seed=args.seed,
            device=args.device
        )
    
    else:
        raise ValueError(f"Algorithme non supporté: {args.algo}")
    
    return model, arch_description


def main():
    """Fonction principale"""
    # Parser les arguments
    args = parse_args()
    
    print("=" * 80)
    print(f"   ENTRAÎNEMENT {args.algo} SUR {args.env}")
    print("=" * 80)
    
    # Créer le dossier d'expérience
    exp_dir = create_experiment_dir(args.algo, args.env)
    print(f"\nDossier d'expérience: {exp_dir}")
    
    # Configurer TensorBoard si activé
    if args.tensorboard:
        tensorboard_log_path = f"{exp_dir}/tensorboard_logs"
        os.makedirs(tensorboard_log_path, exist_ok=True)
        print(f"TensorBoard activé: {tensorboard_log_path}")
    else:
        tensorboard_log_path = None
    
    # Créer les environnements
    print("\nCréation des environnements...")
    env = create_env(args)
    eval_env = create_eval_env(args)
    print(f"  Environnement: {args.env}")
    print(f"  Reward type: {args.reward_type}")
    print(f"  Max episode steps: {args.max_episode_steps}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Créer le modèle
    print(f"\nCréation du modèle {args.algo}...")
    model, arch_description = create_model(args, env, tensorboard_log_path)
    
    # Sauvegarder la configuration (après création du modèle pour avoir arch_description)
    config = vars(args)
    config['network_architecture'] = arch_description  # Ajouter description de l'architecture
    save_config(exp_dir, config)
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Tau: {args.tau}")
    
    # Afficher l'architecture des réseaux
    print(f"\n  🧠 RÉSEAUX DE NEURONES:")
    print(f"    Architecture: {arch_description}")
    print(f"    Activation: {args.activation_fn}")
    
    # Afficher configuration HER
    if args.use_her:
        print(f"\n  🎯 HER ACTIVÉ:")
        print(f"    Strategy: {args.her_goal_selection_strategy}")
        print(f"    N sampled goals: {args.n_sampled_goal}")
        print(f"    → Multiplie les expériences par {args.n_sampled_goal + 1}x")
    else:
        print(f"\n  ⚠️  HER DÉSACTIVÉ (apprentissage difficile avec sparse reward)")
    
    if args.algo == 'TD3':
        print(f"  Policy delay: {args.policy_delay}")
        print(f"  Target policy noise: {args.target_policy_noise}")
    elif args.algo == 'SAC':
        print(f"  Entropy coefficient: {args.ent_coef}")
    
    # Créer le callback d'évaluation
    eval_callback = EvaluationCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_path=f"{exp_dir}/models",
        verbose=args.verbose
    )
    
    # Entraîner
    print(f"\nDébut de l'entraînement ({args.timesteps} timesteps)...")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=eval_callback,
            log_interval=10,
            progress_bar=True
        )
        
        print("\n" + "=" * 80)
        print("✓ Entraînement terminé!")
        print("=" * 80)
        
        # Sauvegarder le modèle final
        final_model_path = f"{exp_dir}/models/final_model"
        model.save(final_model_path)
        print(f"\nModèle final sauvegardé: {final_model_path}")
        
        # Sauvegarder les résultats
        results = {
            'rewards': eval_callback.rewards_history,
            'success_rates': eval_callback.success_history,
            'distances': eval_callback.distance_history,
            'best_mean_reward': eval_callback.best_mean_reward
        }
        save_results(exp_dir, results)
        
        # Tracer les courbes
        if len(eval_callback.rewards_history) > 0:
            plot_training_curves(
                exp_dir,
                eval_callback.rewards_history,
                eval_callback.success_history,
                eval_callback.distance_history
            )
        
        print(f"\n✓ Résultats finaux:")
        print(f"  Meilleur reward moyen: {eval_callback.best_mean_reward:.2f}")
        if len(eval_callback.success_history) > 0:
            print(f"  Dernier taux de succès: {eval_callback.success_history[-1]*100:.1f}%")
        
        # Afficher commande TensorBoard si activé
        if args.tensorboard:
            print(f"\n📊 VISUALISATION TENSORBOARD:")
            print(f"  Commande: tensorboard --logdir {exp_dir}/tensorboard_logs")
            print(f"  Puis ouvrez: http://localhost:6006")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Entraînement interrompu par l'utilisateur")
        # Sauvegarder quand même
        model.save(f"{exp_dir}/models/interrupted_model")
        print(f"Modèle sauvegardé: {exp_dir}/models/interrupted_model")
    
    finally:
        env.close()
        eval_env.close()
    
    print(f"\nTous les fichiers sauvegardés dans: {exp_dir}")


if __name__ == '__main__':
    import numpy as np
    main()
