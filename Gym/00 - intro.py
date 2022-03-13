# -*- coding: utf-8 -*-
"""
title: Gym
subtitle: Intro
source: 
    link: https://gym.openai.com/docs/
file:
    name: gym_00.py
    path: D:\ROBOCZY\Python\Gym\
    date: 2019-03-11 Mon 12:08:21 
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

#%%

env_nam = 'CartPole-v0'
env_nam = 'MountainCar-v0'
env_nam = 'MountainCarContinuous-v0'
env_nam = 'MsPacman-v0'       # needs Atari
env_nam = 'Hopper-v0'
env_nam = 'CarRacing-v0'

#%%

env = gym.make(env_nam)
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
    
#%% better version
    
import gym
env = gym.make(env_nam)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        
#%%
            
import gym
env = gym.make('CartPole-v0')
print(env.action_space)       # Discrete(2)
print(env.observation_space)  # Box(4,)
         
print(env.observation_space.high)   
print(env.observation_space.low)   

#%%

from gym import spaces
space = spaces.Discrete(8)
space
dir(space)
space.n
x = space.sample()
x
assert space.contains(x)

#%%
from gym import envs
print(envs.registry.all())

