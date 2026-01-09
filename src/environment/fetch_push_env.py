"""
Environnement FetchPush Personnalisé
Hérite de Gymnasium FetchPush-v2 mais avec contrôle total sur:
- Observations
- Actions  
- Reward function
- Physique
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv


class CustomFetchPushEnv(MujocoFetchPushEnv):
    """
    Environnement FetchPush personnalisé.
    
    Vous pouvez modifier:
    1. _get_obs() - Pour changer les observations
    2. step() - Pour modifier les actions et la physique
    3. compute_reward() - Pour personnaliser la reward function
    4. _sample_goal() - Pour changer comment les cibles sont générées
    """
    
    def __init__(
        self,
        reward_type="dense",
        max_episode_steps=200,
        **kwargs
    ):
        # Configuration personnalisée
        self.custom_reward_type = reward_type
        self.max_steps = max_episode_steps
        self.current_step = 0
        
        # Variables pour rewards robustes (progression, potential)
        self.prev_distance_to_target = None
        self.prev_distance_to_object = None
        self.initial_distance_to_target = None
        self.best_distance_to_target = float('inf')
        
        # Initialiser l'environnement parent
        super().__init__(reward_type=reward_type, **kwargs)
        
        # Modifier l'observation space pour les 17 observations optimisées
        # Ancienne: observation (25D) → Nouvelle: observation (17D)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            ),
            'achieved_goal': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
            ),
            'desired_goal': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
            ),
        })
        
        # Garder l'action space à 4D pour compatibilité avec l'environnement parent
        # Le gripper sera forcé à fermé (-1.0) dans _customize_action
        # L'agent contrôle [dx, dy, dz, gripper] mais gripper est ignoré
        
        print(f"[CustomFetchPush] Initialized")
        print(f"  Reward type: {reward_type}")
        print(f"  Max episode steps: {max_episode_steps}")
        print(f"  Observation space: observation(17D) + achieved_goal(3D) + desired_goal(3D)")
        print(f"  Action space: Box(4,) - [dx, dy, dz, gripper] (gripper forcé à fermé)")
    
    def reset(self, **kwargs):
        """Reset avec compteur de steps"""
        self.current_step = 0
        obs, info = super().reset(**kwargs)
        
        # Réinitialiser les variables pour rewards robustes
        distance_to_target = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        self.prev_distance_to_target = distance_to_target
        self.initial_distance_to_target = distance_to_target
        self.best_distance_to_target = distance_to_target
        
        grip_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        self.prev_distance_to_object = np.linalg.norm(grip_pos - object_pos)
        
        # Personnaliser l'observation
        obs = self._customize_observation(obs)
        
        return obs, info
    
    def step(self, action):
        """
        Step avec personnalisation des actions et reward.
        
        Args:
            action: Array de shape (4,) avec [dx, dy, dz, gripper]
                    dx, dy, dz: Déplacement du gripper (-1 à 1, échelle ~5cm)
                    gripper: Contrôle des doigts (-1=fermé, 1=ouvert)
        
        Returns:
            observation: Dict avec 'observation', 'achieved_goal', 'desired_goal'
            reward: Float
            terminated: Bool (True si objectif atteint)
            truncated: Bool (True si max steps)
            info: Dict avec métadonnées
        """
        self.current_step += 1
        
        # PERSONNALISATION 1: Modifier l'action si nécessaire
        action = self._customize_action(action)
        
        # Exécuter l'action dans la simulation MuJoCo
        obs, reward, terminated, truncated, info = super().step(action)
        
        # PERSONNALISATION 2: Modifier l'observation
        obs = self._customize_observation(obs)
        
        # PERSONNALISATION 3: Reward personnalisée
        reward = self._customize_reward(obs, info)
        
        # Mettre à jour les variables pour le prochain step
        distance_to_target = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        grip_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        distance_to_object = np.linalg.norm(grip_pos - object_pos)
        
        self.prev_distance_to_target = distance_to_target
        self.prev_distance_to_object = distance_to_object
        self.best_distance_to_target = min(self.best_distance_to_target, distance_to_target)
        
        # Ajouter la limite de steps
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Ajouter des infos personnalisées
        info.update(self._get_custom_info(obs))
        
        return obs, reward, terminated, truncated, info
    
    def _customize_observation(self, obs):
        """
        OBSERVATIONS OPTIMISÉES (17D)
        
        Structure:
        - observation (17D):
            [0:3]   grip_pos: Position du gripper (X, Y, Z)
            [3:6]   object_pos: Position de l'objet (X, Y, Z)
            [6:9]   desired_goal: Position cible (X, Y, Z)
            [9:12]  grip_velp: Vélocité du gripper (vX, vY, vZ)
            [12:15] object_velp: Vélocité de l'objet (vX, vY, vZ)
            [15]    distance_to_object: Distance gripper → objet
            [16]    distance_to_target: Distance objet → cible
        
        - achieved_goal (3D): Position actuelle de l'objet (pour compute_reward)
        - desired_goal (3D): Position cible (pour compute_reward)
        
        Observations RETIRÉES (inutiles pour pousser):
        - object_rel_pos: Redondant (= object_pos - grip_pos)
        - gripper_state: Inutile (on ne saisit pas l'objet)
        - object_rot: Rotation non importante pour pousser
        - object_velr: Vélocité angulaire non nécessaire
        - gripper_vel: Inutile si gripper reste ouvert
        """
        
        # Extraire les infos de l'observation originale (25D)
        original_obs = obs['observation']
        
        # Positions
        grip_pos = original_obs[0:3]       # Position du gripper
        object_pos = original_obs[3:6]     # Position de l'objet
        
        # Vélocités
        object_velp = original_obs[15:18]  # Vélocité linéaire de l'objet
        grip_velp = original_obs[21:24]    # Vélocité du gripper
        
        # Cibles (déjà dans obs)
        desired_goal = obs['desired_goal']
        
        # Calculer les distances clés
        distance_to_object = np.linalg.norm(grip_pos - object_pos)
        distance_to_target = np.linalg.norm(object_pos - desired_goal)
        
        # Construire la nouvelle observation (17D)
        new_observation = np.concatenate([
            grip_pos,                    # [0:3]   Position gripper
            object_pos,                  # [3:6]   Position objet
            desired_goal,                # [6:9]   Position cible
            grip_velp,                   # [9:12]  Vélocité gripper
            object_velp,                 # [12:15] Vélocité objet
            [distance_to_object],        # [15]    Distance gripper → objet
            [distance_to_target],        # [16]    Distance objet → cible
        ])
        
        # Mettre à jour l'observation
        obs['observation'] = new_observation.astype(np.float64)
        
        return obs
    
    def _customize_action(self, action):
        """
        ACTIONS OPTIMISÉES (4D avec gripper fixé)
        
        L'agent envoie des actions 4D:
        - action[0]: dx - Déplacement horizontal X (-1 à 1)
        - action[1]: dy - Déplacement horizontal Y (-1 à 1)  
        - action[2]: dz - Déplacement vertical Z (-1 à 1, limité à ±20%)
        - action[3]: gripper - IGNORÉ (sera forcé à -1.0 = fermé)
        
        Cette fonction modifie l'action avant de la passer à l'environnement parent.
        
        Pourquoi gripper fermé ?
        - Contact précis et contrôlé
        - Évite que l'objet se coince entre les doigts
        - Force de poussée mieux transmise
        """
        
        # S'assurer que l'action est bien 4D
        assert action.shape == (4,), f"Action doit être 4D, reçu: {action.shape}"
        
        # Clipper les actions
        action = np.clip(action, -1.0, 1.0)
        
        # Créer l'action modifiée
        modified_action = np.zeros(4, dtype=np.float32)
        
        # Mouvements horizontaux (inchangés)
        modified_action[0] = action[0]  # dx
        modified_action[1] = action[1]  # dy
        
        # Mouvement vertical (LIMITÉ à 20% pour stabilité)
        modified_action[2] = action[2]  # dz réduit
        
        # Gripper FERMÉ (fixé, on ignore action[3])
        modified_action[3] = -1.0  # -1 = fermé
        
        return modified_action
    
    def _customize_reward(self, obs, info):
        """
        REWARD FUNCTIONS ROBUSTES
        
        Types disponibles:
        - sparse: Binary reward (0 ou -1)
        - dense: Distance normalisée et clippée
        - dense_normalized: Distance normalisée dans [-1, 0]
        - two_stage: Approcher objet puis pousser vers cible
        - shaped: Multi-critères équilibrés
        - potential_based: Reward shaping théorique (policy-invariant)
        """
        
        # Extraire les informations
        achieved = obs['achieved_goal']
        desired = obs['desired_goal']
        grip_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        
        distance_to_target = np.linalg.norm(achieved - desired)
        distance_to_object = np.linalg.norm(grip_pos - object_pos)
        
        # === SPARSE ===
        if self.custom_reward_type == 'sparse':
            return 0.0 if info.get('is_success', False) else -1.0
        
        # === DENSE (simple mais clipé) ===
        elif self.custom_reward_type == 'dense':
            # Clipper pour éviter les valeurs extrêmes
            reward = -distance_to_target
            reward = np.clip(reward, -1.0, 0.0)
            return reward
        
        # === DENSE NORMALIZED (plus robuste) ===
        elif self.custom_reward_type == 'dense_normalized':
            # Normaliser par la distance initiale
            if self.initial_distance_to_target > 0:
                normalized_distance = distance_to_target / self.initial_distance_to_target
            else:
                normalized_distance = 0.0
            
            # Reward dans [-1, 0]
            reward = -normalized_distance
            
            # Bonus de succès
            if info.get('is_success', False):
                reward += 1.0  # Reward final = 0.0
            
            return reward
        
        # === TWO STAGE (approcher puis pousser) ===
        elif self.custom_reward_type == 'two_stage':
            reward = 0.0
            
            # Stage 1: Approcher l'objet (si loin)
            if distance_to_object > 0.05:  # Pas encore en contact
                reward = -distance_to_object * 2.0  # Poids x2 pour priorité
            
            # Stage 2: Pousser vers la cible (si proche de l'objet)
            else:
                reward = -distance_to_target * 3.0  # Poids x3 (objectif principal)
                reward += 0.5  # Bonus constant pour contact
            
            # Bonus de succès
            if info.get('is_success', False):
                reward += 10.0
            
            return np.clip(reward, -10.0, 10.0)
        
        # === SHAPED (multi-critères équilibrés) ===
        elif self.custom_reward_type == 'shaped':
            reward = 0.0
            
            # 1. Terme principal: distance à la cible (poids fort)
            reward -= distance_to_target * 2.0
            
            # 2. Encourager le contact avec l'objet
            if distance_to_object < 0.05:
                reward += 0.3
            else:
                reward -= distance_to_object * 0.5
            
            # 3. Bonus de progression (récompenser amélioration)
            if self.prev_distance_to_target is not None:
                improvement = self.prev_distance_to_target - distance_to_target
                reward += improvement * 5.0  # Fort bonus pour progression
            
            # 4. Bonus de succès
            if info.get('is_success', False):
                reward += 10.0
            
            # 5. Pénalité temporelle légère (efficacité)
            reward -= 0.01
            
            # Clipper pour stabilité
            return np.clip(reward, -5.0, 15.0)
        
        # === POTENTIAL BASED (théoriquement optimal) ===
        elif self.custom_reward_type == 'potential_based':
            # Fonction de potentiel: Φ(s) = -(distance_to_target + 0.5 * distance_to_object)
            gamma = 0.95  # Discount factor
            
            # Potentiel actuel
            potential_current = -(distance_to_target + 0.5 * distance_to_object)
            
            # Potentiel précédent
            if self.prev_distance_to_target is not None:
                potential_prev = -(self.prev_distance_to_target + 0.5 * self.prev_distance_to_object)
            else:
                potential_prev = potential_current
            
            # Reward shaping: R' = R + γ*Φ(s') - Φ(s)
            base_reward = 0.0 if info.get('is_success', False) else -1.0
            shaping = gamma * potential_current - potential_prev
            
            reward = base_reward + shaping
            
            # Bonus massif pour succès
            if info.get('is_success', False):
                reward += 50.0
            
            return reward
        
        else:
            # Défaut: sparse
            return 0.0 if info.get('is_success', False) else -1.0
    
    def _get_custom_info(self, obs):
        """
        Ajouter des informations supplémentaires dans le dict info
        """
        grip_pos = obs['observation'][:3]
        object_pos = obs['achieved_goal']
        target_pos = obs['desired_goal']
        
        return {
            'step': self.current_step,
            'distance_to_target': np.linalg.norm(object_pos - target_pos),
            'distance_to_object': np.linalg.norm(grip_pos - object_pos),
            'gripper_height': grip_pos[2],
        }
    
    def _sample_goal(self):
        """
        PERSONNALISATION DE LA GÉNÉRATION DE CIBLES
        
        Par défaut, la cible est sur la table de façon aléatoire.
        Vous pouvez:
        1. Fixer la position de la cible
        2. Générer des cibles plus difficiles
        3. Créer un curriculum learning
        """
        
        # Appeler la méthode parent
        goal = super()._sample_goal()
        
        # Exemple: Fixer la cible à une position spécifique
        # goal = np.array([1.3, 0.75, 0.425])  # Position fixe
        
        # Exemple: Cible plus loin (plus difficile)
        # goal[:2] *= 1.5  # Augmenter la distance
        
        return goal
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute reward (appelé par l'environnement parent)
        Redirects vers _customize_reward
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        
        if self.custom_reward_type == 'sparse':
            return -(distance > 0.05).astype(np.float32)
        else:
            return -distance


def make_fetch_push_env(
    reward_type='dense',
    max_episode_steps=200,
    render_mode=None
):
    """
    Factory function pour créer l'environnement personnalisé.
    
    Args:
        reward_type: Type de reward function
            - 'sparse': Binary (0 ou -1)
            - 'dense': Distance normalisée et clippée [-1, 0]
            - 'dense_normalized': Distance normalisée par distance initiale
            - 'two_stage': Approcher puis pousser
            - 'shaped': Multi-critères équilibrés
            - 'potential_based': Reward shaping théorique
        max_episode_steps: Nombre maximum de steps par épisode
        render_mode: 'human' pour visualisation, None sinon
    
    Returns:
        CustomFetchPushEnv instance
    """
    return CustomFetchPushEnv(
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode
    )
