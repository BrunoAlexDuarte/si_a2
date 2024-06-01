
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

class DigDugEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(48,24), dtype=np.int32, minimum=0, name='observation')
    self._enemies = None; #Posicao de todos os inimigos
    self._last_action = None; #Ultima acao
    self._close = None; #Mapa da proximidade ao digdug
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = np.zeros((48,24), np.int32);
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def init_state():
      self._enemies = np.array();
      self._close = np.array();
      self._last_action = None; 
      return ts.restart(); #Verificar bem esta função

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 0: #Don't do anything
      self._episode_ended = True
    elif action == 1: #Up
        print("UP");
    elif action == 2: #Down
        print("DOWN");
    elif action == 3: #Left
        print("LEFT");
    elif action == 4: #Right
        print("RIGHT");
    elif action == 5: #Attack
        print("ATTACK");


    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

    def _closest_enemy(self):
        # Calcula a distância euclidiana entre o jogador e o inimigo mais próximo
        player_pos = np.array(self.player_pos)
        min_distance = float(100.0)
        enemy = [0,0]
        for enemy_pos in self.enemies:
            enemy_pos = np.array(enemy_pos)
            distance = abs(player_pos[0]-enemy_pos[0])+abs(player_pos[1]-enemy_pos[1])
            if distance < min_distance:
                min_distance = distance
                enemy = enemy_pos
        return enemy


get_new_card_action = np.array(0, dtype=np.int32)
end_round_action = np.array(1, dtype=np.int32)

environment = DigDugEnv()
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(3):
  time_step = environment.step(get_new_card_action)
  print(time_step)
  cumulative_reward += time_step.reward

time_step = environment.step(end_round_action)
print(time_step)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)
print('State = ', environment.get_state())

