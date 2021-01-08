import numpy as np
import gym
import rl

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

def define_layers(env, nb_actions):
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(nb_actions, activation='linear'))
  return model