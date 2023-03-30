import os, sys
import gymnasium as gym
import time
import pickle

import text_flappy_bird_gym

from src import sarsa_agent, q_agent, utils

if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    obs, info = env.reset()

    agent = q_agent.QLearningAgent()

    with open('models/q_learning.pkl', 'rb') as pickle_file:
        Q = pickle.load(pickle_file)

    agent_info = {"num_actions": 2, "epsilon": 0.1, "step_size": 0.7, "discount": 1.0}
    agent_info["seed"] = 0
    agent_info["policy"] = Q
    agent.agent_init(agent_info)

    # iterate
    while True:

        # Select next action
        # action = env.action_space.sample()  
        action = agent.policy(obs)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.2) # FPS

        # If player is dead break
        if done:
            break

    env.close()
