import gym
from gym import spaces
import numpy as np
import os
import websockets
import asyncio
import json

async def reset_game():
    print("In function reset")
    async with websockets.connect(f"ws://localhost:8000/player") as websocket:
    # Receive information about static game properties
        print("Sent reset")
        await websocket.send(json.dumps({"cmd": "reset"}))

class DigDugEnv(gym.Env):
    def __init__(self):
        super(DigDugEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Attack
        self.observation_space = spaces.Box(low=0, high=1, shape=(41,1), dtype=np.float32)
        self.state = self.reset()
        self.player_pos = [0,0]
        self.enemies = []
        self._count = 0
        self._reset = False

    def is_reset(self):
        return self._reset
    def set_reset(self):
        self._reset = False
        
        
    def reset(self):
        print("Bigger reset")
        self.state = np.zeros((41, 1))
        #The initial position
        self.player_pos = [0,0]
        #No enemies detected at the beginning 
        self.enemies = []
        self._reset = True
        return self.state
    
    def _get_observation(self,map,digdug,target,last_action,direction, num_enemies):
        #Colocamos na matriz do mapa 1 caso tenha o dd e 2 caso tenha um inimigo
         #Eventualmente podemos diferenciar o tipo de inimigo
        #Colocar se um inimigo morreu na ronda anterior, ou se foi atingido
        observation = np.zeros((41,1))
        observation[40] = direction
        observation[39] = last_action
        print("DD:", digdug)
        if target[0] != -1:
            observation[37] = target[0] - digdug[0]
            observation[38] = target[1] - digdug[1]
        else:
            observation[37] = -1
            observation[38] = -1
        observation[35] = digdug[0]
        observation[36] = digdug[1]
    
        if direction == 0: #Up
            ddx = digdug[0] - 2
            ddy = digdug[1] + 2
            for i_1 in range(7):
                for i_2 in range(5):
                    observation[i_1*5 + i_2] = self.get_from_map(map, ddx + i_2, ddy - i_1) 
        elif direction == 1: #Down
            ddx = digdug[0] + 2
            ddy = digdug[1] - 2
            for i_1 in range(7):
                for i_2 in range(5):
                    observation[i_1*5 + i_2] = self.get_from_map(map, ddx - i_2, ddy + i_1) 
        elif direction == 2:  #Left
            ddx = digdug[0] + 2
            ddy = digdug[1] + 2
            for i_1 in range(7):
                for i_2 in range(5):
                    observation[i_1*5 + i_2] = self.get_from_map(map, ddx - i_1, ddy - i_2)  
        else: #Right
            ddx = digdug[0] - 2
            ddy = digdug[1] - 2
            for i_1 in range(7):
                for i_2 in range(5):
                    observation[i_1*5 + i_2] = self.get_from_map(map, ddx + i_1, ddy + i_2)  
        return observation

    def get_from_map(self, map, posx, posy):
        if posx < 0 or posx > 47 or posy < 0 or posy > 23:
            return -1
        return map[posx][posy]

    def step(self, action):
        closest_enemy = self._closest_enemy()
        dx = closest_enemy[0] - self.player_pos[0]
        dy = closest_enemy[1] - self.player_pos[1]

        prev_distance = abs(dx) + abs(dy)

        self._count += 1
        if self._count == 300:
            self.reset()
            self._count = 0

        reward = -1  # Small penalty for each step to encourage faster actions

        if action == 0:  # Up
            if self.player_pos[0] < 47:
                self.player_pos[0] += 1
        elif action == 1:  # Down
            if self.player_pos[0] > 0:
                self.player_pos[0] -= 1
        elif action == 2:  # Left
            if self.player_pos[1] > 0:
                self.player_pos[1] -= 1
        elif action == 3:  # Right
            if self.player_pos[1] < 23:
                self.player_pos[1] += 1
        elif action == 4:  # Attack
            if abs(dy) < 3 and abs(dx) < 3:
                reward = 300  # High reward for a successful attack
            else:
                reward = - (abs(dx) + abs(dy))

        # Recalculate distance after the action
        new_dx = closest_enemy[0] - self.player_pos[0]
        new_dy = closest_enemy[1] - self.player_pos[1]
        new_distance = abs(new_dx) + abs(new_dy)

        # Provide positive reward if the agent moves closer to the enemy
        if new_distance < prev_distance:
            reward += 10  # Reward for moving closer to the enemy
        elif new_distance > prev_distance:
            reward -= 10  # Penalty for moving away from the enemy

        next_state = self.state
        done = False

        return next_state, reward, done, {}

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
