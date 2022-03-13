#! python3
"""
title: Deep RL and Controls OpenAI Gym Recitation
subtitle: 
keywords: [Gym, OpenAI, Reinforcement Learning]
todo:
    - investigate virtualenv
sources:   
    - title: Deep RL and Controls OpenAI Gym Recitation
      link: "D:\bib\Python\OpenAI Gym Recitation (43sl).pdf"
      authors: 
          - nick: 
            fullname: Devin Schwab
            email: 
file:
    name: "02 - Devin Shwab.py"
    path: D:\ROBOCZY\Python\Gym
    date: 2019-03-21
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp@interia.pl
              - akasp666@google.com
              - arek@staart.pl
"""              

#%%  in console

virtualenv openai-gym-demo
source openai-gym-demo/bin/activate
pip install -U gym[all]
python -c 
import gym; gym.make("FrozenLake-v0")


#%%  Basic Agent Loop


import gym

env = gym.make("Taxi-v2")     # https://gym.openai.com/envs/Taxi-v2/
state_0 = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    state_, reward, done, info = env.step(action)
    if done:
        env.render()
        break
    
#%% action and state space - 

dir(gym.spaces)
    
dir(gym.spaces.Discrete)
dir(gym.spaces.MultiDiscrete)
dir(gym.spaces.Box)
dir(gym.spaces.Tuple)

#%%

dten = gym.spaces.Discrete(10)
dten
dir(dten)
dten.contains(0)
dten.contains(9)
dten.contains(10)
dten.shape

[dten.sample() for k in range(10)]


#%%

import numpy as np

mdisc = gym.spaces.MultiDiscrete([1, 3, 5])
dir(mdisc)
mdisc.contains([1, 1, 1])   # ERROR!
mdisc.contains(np.array([1, 1, 1]))   # False

mdisc = gym.spaces.MultiDiscrete([2, 3, 5])
mdisc.contains(np.array([1, 1, 1]))   # True

mdisc.shape   # (3,)
[mdisc.sample() for k in range(10)]
[print(mdisc.sample()) for k in range(10)]


#%%
"""
WARNING! There's NO input data validity check!!!
"""
mdisc = gym.spaces.Discrete([(1, 3), (0, 5)])   # it works but it shouldn't!!!
mdisc.sample()  #! ERROR
mdisc.contains(2)  #! ERROR
mdisc.contains([2, 1])   # False ...  -- nonsense!


#%%

box = gym.spaces.Box(np.array((-1., 1., 0)),   # low
                     np.array((0., 2., 2.)),    # high
                     dtype=float)  
# multidimensional continuous spaces with bounds

dir(box)
box.high
box.low
box.contains(np.array((-.1, 1.1)))   # False
box.contains(np.array((-.1, 1.1, 1.)))  # True

[box.sample() for _ in range(10)]

#%%

tup = gym.spaces.Tuple((mdisc, box))
dir(tup)
tup.sample()
[tup.sample() for _ in range(10)]

#%%

mdisc = gym.spaces.MultiDiscrete([2, 3, 5])
mdisc.contains(np.array([1, 1, 1]))   # True

box = gym.spaces.Box(np.array((-1., 1., 0)),   # low
                     np.array((0., 2., 2.)),    # high
                     dtype=float) 
box.contains(np.array((-.1, 1.1, 1.)))  # True

tup = gym.spaces.Tuple((mdisc, box))
tup.contains((np.array([1, 1, 1]), np.array((-.1, 1.1))))      # False
tup.contains((np.array([1, 1, 1]), np.array((-.1, 1.1, 1.))))  # True

#%%
""" Creating new environments
"""

...


#%%  Monitor Wrapper

cd d:/ROBOCZY/Python/Gym

#%%
import gym
import ffmpeg
from gym import wrappers
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './cartpole-experiment-1')
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
env.close()
gym.upload('./cartpole-experiment-1', api_key='blah')

#!  ERROR: DependencyNotInstalled:f

#%%

