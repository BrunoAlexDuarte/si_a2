"""Example client."""
import asyncio
import getpass
import json
import os
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.policies import policy_saver
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential
import tf_keras
from tf_agents.policies import random_tf_policy
import gym
from gym import spaces

import websockets
import envg




def closer_enemy(player_pos, enemies):
    # Calcula a distância euclidiana entre o jogador e o inimigo mais próximo
    min_distance = float(100.0)
    enemy = [0,0]
    if len(enemies) == 0:
        print("No enemies")
        return [-1, -1]
    print("ENEMIES:", enemies)
    for enemy_pos in enemies: 
        enemy_pos = np.array(enemy_pos)
        distance = abs(player_pos[0]-enemy_pos[0])+abs(player_pos[1]-enemy_pos[1])
        if distance < min_distance:
            min_distance = distance
            enemy = enemy_pos
    return enemy

env = gym_wrapper.GymWrapper(envg.DigDugEnv())
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)

fc_layer_params = (100,50)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer = optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=10000
)

# Dataset setup for training
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2
).prefetch(3)

iterator = iter(dataset)

#policy_dir = 'policy'
#policy_saver.PolicySaver(agent.policy).save(policy_dir)

async def agent_loop(server_address="localhost:8000", agent_name="student"):
    """Example client loop."""
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        # Receive information about static game properties
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        if os.path.exists('policy'):
            policy_s= tf.compat.v2.saved_model.load('policy')
            print("Loaded saved policy.")
        else:
            policy_s = None
        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
        #policy = tf.compat.v2.saved_model.load(policy_dir)
        time_step = train_env.reset()
        t = 0
        initial_exp = 100
        initial_p = 1500
        epsilon = 0.1

        last_action = None
        last_direction = 3
        mapa = np.zeros((48,24))
        while True:
            try:
                state = json.loads(
                    await websocket.recv()
                )  # receive game update, this must be called timely or your game will get out of sync with the server
                key = ""
                action = 0
                t+=1
                digdug = state.get("digdug", [1, 1])
                enemies = []
                enemy_info_all = state.get("enemies", [])
                for enemy_info in enemy_info_all:
                    enemies.append(enemy_info["pos"])
                num_enemies = len(enemies)
                target = closer_enemy(digdug, enemies)
                if state.get('map'):
                    mapa = state.get('map')
                observation = env._get_observation(mapa, digdug, target, last_action, last_direction, num_enemies)

                time_step = train_env.current_time_step()

                # action_spec = train_env.action_spec()
                # action = np.random.randint(action_spec.minimum, action_spec.maximum + 1)
                if t < initial_p:
                    action_step = random_policy.action(time_step)
                elif initial_exp <= t < initial_p:
                    if np.random.rand() < epsilon:
                        action_step = random_policy.action(time_step)
                    else:
                        action_step = agent.policy.action(time_step)
                else:
                    if policy_s:
                        action_step = policy_s.action(time_step)
                    else:
                        action_step = agent.policy.action(time_step)
                    
                action = action_step.action.numpy()[0]
                last_action = action
                if action != 4:
                    last_direction = action
                print(action)
                if action == 0:
                    key = 'w'
                elif action == 1:
                    key = 's'
                elif action == 2:
                    key = 'a'
                elif action == 3:
                    key = 'd'
                elif action == 4:
                    key = 'A'
                    
                if env.is_reset():
                    env.set_reset()
                    await websocket.send(json.dumps({"cmd": "reset"}))


                await websocket.send(
                    json.dumps({"cmd": "key", "key": key})
                )  # send key command to server - you must implement this send in the AI agent
                
                digdug = state.get("digdug", [1,1])
                enemies = []
                enemy_info_all = state.get("enemies", [])
                for enemy_info in enemy_info_all:
                    enemies.append(enemy_info["pos"])
                num_enemies = len(enemies)
                target = closer_enemy(digdug, enemies)

                observation = env._get_observation(mapa, digdug, target, last_action, last_direction, num_enemies)
                next_time_step = train_env.step(action)
                
                traj = trajectory.from_transition(time_step,action_step,next_time_step)
                replay_buffer.add_batch(traj)

                time_step = next_time_step
                if t>= initial_exp:
                    experience, unused_info = next(iterator)
                    agent.train(experience)
            except websockets.exceptions.ConnectionClosedOK:
                print("Server has cleanly disconnected us")
                return

         
# DO NOT CHANGE THE LINES BELLOW
# You can change the default values using the command line, example:
# $ NAME='arrumador' python3 client.py
loop = asyncio.get_event_loop()
SERVER = os.environ.get("SERVER", "localhost")
PORT = os.environ.get("PORT", "8000")
NAME = os.environ.get("NAME", getpass.getuser())
loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))
policy_saver.PolicySaver(agent.policy).save('policy')
print("saved policy")
