# -*- coding: utf-8 -*-
"""
title: FrozenLake8x8-v0 example
subtitle: 
keywords: [AI/ML, Reinforcement Learning, OpenAI, Gym]
source: 
    title: Reinforcement Learning with OpenAI Gym - Introduction
    link: https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
    date: 2018-09-21
    author: 
        - nick: 
          fullname: Ashish Rana
          email: 
file:
    name: gym_01.py
    path: D:\ROBOCZY\Python\Gym\
    date: 2019-03-12 Tue 16:30:00
    author: 
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@tieto.com
              - akasp@interia.pl
              - akasp666@gmail.com
              - arek@staart.pl
"""              

#%%

import gym
import numpy as np

#%% 

env = gym.make('FrozenLake8x8-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])

#%% parameters of Q-learning

eta = .628
gamma = .9
episodes = 5000
r_sum_list = []

#%% algorithm

for i in range(episodes):
    s = env.reset()
    r_sum = 0
    done = False
    j = 0
    # learning
    while j < 99:
        env.render()
        j += 1
        # choose action from Q-table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        s1, r, done, info = env.step(a)
        # update Q-table
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.max(Q[s1, a]) - Q[s, a])
        r_sum += r
        s = s1
        if done:
            break
    r_sum_list.append(r_sum)
    env.render()



#%% 

#%%
