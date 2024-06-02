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
        self.observation_space = spaces.Box(low=0, high=1, shape=(48, 24, 1), dtype=np.float32)
        self.state = self.reset()
        self.player_pos = [0,0]
        self.enemies = []
        self._count = 0;
        self._reset = False;

    def is_reset(self):
        return self._reset;
    def set_reset(self):
        self._reset = False;
        
        
    def reset(self):
        print("Bigger reset")
        self.state = np.zeros((48, 24, 1))
        #The initial position
        self.player_pos = [0,0]
        #No enemies detected at the beginning 
        self.enemies = []
        self._reset = True;
        return self.state
    
    def _get_observation(self,enemies_pos):
        #Colocamos na matriz do mapa 1 caso tenha o dd e 2 caso tenha um inimigo
         #Eventualmente podemos diferenciar o tipo de inimigo
        #Colocar se um inimigo morreu na ronda anterior, ou se foi atingido
        observation = np.zeros((48,24))
        self.enemies = enemies_pos
        px, py = self.player_pos
        px = max(0, min(px, 47))  
        py = max(0, min(py, 23)) 
        observation[px, py] = 1
        for x, y in self.enemies:
            x = max(0, min(x, 47))  
            y = max(0, min(y, 23))  
            observation[x, y] = 2

        return observation

    def step(self, action):
        closest_enemy =self._closest_enemy()
        dx =closest_enemy[0] - self.player_pos[0]
        dy = closest_enemy[1] - self.player_pos[1]
        #print(self.player_pos)
        #print(closest_enemy)
        print("count:", self._count)
        self._count += 1;

        if self._count == 300:
            self.reset();
            self._count = 0;


        if action == 0: #Up
            if self.player_pos[0] < 47:
                self.player_pos[0]+=1
            #The reward will be negative to make the agent take the least steps to a destination
            if dx > 0 and abs(dx) > 3:
                reward = - (abs(dx) + abs(dy)) + 100 #If it goes in the right direction the penalty will be less
            else:
                reward = - (abs(dx) + abs(dy))
        elif action == 1: #Down
            if self.player_pos[0] > 0:
                self.player_pos[0]-=1
            #The reward will be negative to make the agent take the least steps to a destination
            if dx < 0 and abs(dx) > 3:
                reward = - (abs(dx) + abs(dy)) + 100 #If it goes in the right direction the penalty will be less
            else:
                reward = - (abs(dx) + abs(dy))
        elif action == 2: #Left
            if self.player_pos[1] > 0:
                self.player_pos[1] -=1
            #The reward will be negative to make the agent take the least steps to a destination
            if dy < 0 and abs(dy)>3: 
                reward = - (abs(dx) + abs(dy)) + 100 #If it goes in the right direction the penalty will be less
            else:
                reward = - (abs(dx) + abs(dy))
        elif action == 3: #Right
            if self.player_pos[1] < 23:
                self.player_pos[1] +=1
            #The reward will be negative to make the agent take the least steps to a destination
            if dy > 0 and abs(dy)>3: 
                reward = - (abs(dx) + abs(dy)) + 100  #If it goes in the right direction the penalty will be less
            else:
                reward = - (abs(dx) + abs(dy))
        elif action == 4: #Attack
            if abs(dy) < 3 and abs(dx) < 3:
                reward = 300
            else:
                reward = - (abs(dx) + abs(dy))

        #Colocar estado para se o inimigo morreu ganhar mais premio
                
        next_state = self.state
        
        #print(reward)
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





