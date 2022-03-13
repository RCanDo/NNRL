# -*- coding: utf-8 -*-
"""
title: CartPole-v0 example
subtitle: 
keywords: [Gym, OpenAI, Deep Learning, Reinforcement Learning, AI/ML]
source: 
    - link: https://gym.openai.com/envs/CartPole-v1/
    - title: Using Keras Reinforcement Learning API with OPENAI GYM
      link: https://medium.com/@abhishek.bn93/using-keras-reinforcement-learning-api-with-openai-gym-6c2a35036c83
      date: 2019-02-10
      author:
          - nick:
            fullname: Abhishek Nagaraja
            email: 
file:
    name: 
    path: 
    date: 2019-03-14 Thu 10:30:00
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
import random
import numpy as np
import rl   # pip install keras-rl
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

#%%

env = gym.make('CartPole-v1')

dir(env.observation_space)
env.observation_space.shape  # dimension of observation space
env.observation_space.high
env.observation_space.low
env.observation_space.sample()
env.observation_space.np_random  # ?


env.action_space.n  # nr of possible actions

#%%

episodes = range(20)

for _ in episodes:
    state = env.reset()
    done = False
    score = 0
    i = 0
    while not done:
        env.render()
        action = random.choice([0, 1])
        state, reward, done, info = env.step(action)
        score += reward
        i += 1
    print("episode {:2} - iters {:2} - score {:4.1f}".format(_, i, score))
    

#%%
#%% Q-table
def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='relu'))
    return model

model = agent(env.observation_space.shape[0], env.action_space.n)

model.summary()
    

#%%
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

policy = EpsGreedyQPolicy()

sarsa = SARSAAgent(model=model,                 
                   policy=policy, 
                   nb_actions=env.action_space.n   # from env !
                  )

sarsa.compile('adam', metrics=['mse'])   # just model.compile(...)

sarsa.fit(env, nb_steps=5e4, visualize=False, verbose=1)

#%%
scores = sarsa.test(env, nb_episodes=100, visualize=False)
mean_score = np.mean(scores.history['episode_reward'])
print('Average score over 100 test games: {}'.format(mean_score))

#%%
sarsa.save_weights('sarsa_weights.h5f', overwrite=True)

#%%
sarsa.load_weights('sarsa_weights.h5f')

#%%
# how the trained agent works

sarsa.test(env, nb_episodes=2, visualize=True)

#%%

sarsa.get_config()


