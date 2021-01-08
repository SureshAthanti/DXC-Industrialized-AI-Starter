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

def define_policy(model_name):
  policy = []
  if (model_name == "DQN"):
    policy = BoltzmannQPolicy()
  return policy