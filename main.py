import gym
import numpy as np
import time
import os
import random
from agent import Agent


if __name__ == "__main__":
        
    env = gym.make("FrozenLake-v0")
    episodes = 50000
    steps_in_episode = 100
    learning_rate = 0.1
    discount_rate = 0.99
    epsilon_decay_rate = 0.0005
    min_epsilon_value = 0.01
    agent = Agent(16, env.action_space.n, learning_rate, discount_rate, epsilon_decay_rate, min_epsilon_value)

    rewards = []
    steps = []
    for episode in range(episodes):
        # reset env
        state = env.reset()

        # run episode
        done = False
        for step in range(steps_in_episode):
            # get action from agent
            action = agent.choose_action(state, episode)

            # get new state and reward from simulation
            new_state, reward, done, info = env.step(action)

            # update agents q table
            agent.update_q_table(state, action, new_state, reward)
            state = new_state

            if done:
                rewards.append(reward)
                steps.append(step)
                break
        else:
            rewards.append(reward)
            steps.append(step)

        if episode % 1000 == 0:
            print(f"--Episode {episode}--")
            env.render()
            print(f"Average reward: {np.average(rewards)} steps: {np.average(steps)} epsilon: {agent.epsilon}\n")
            rewards = []
            steps = []
        

    print("Done training")
    print(agent.q_table)
    if input("render?"):
        for episode in range(10):
            state = env.reset()
            done = False
            for frame in range(100):
                os.system("clear")
                env.render()
                agent.epsilon = 0
                action = agent.choose_action(state)
                print(episode, frame)
                state, reward, done, _ = env.step(action)
                time.sleep(0.1)
                if done:
                    if reward == 1:
                        env.render()
                        time.sleep(5)
                    break



