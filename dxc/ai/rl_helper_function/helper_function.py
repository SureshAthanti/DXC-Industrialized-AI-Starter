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

def Main_Helper(env_name, model_name):

  nb_actions = 0
  if (model_name == "DQN"):
    nb_actions = env.action_space.n
  elif (model_name == "DDPG"):
    nb_actions = env.action_space.shape[0]


  model_structure = define_layers(env, nb_actions)
  memory = define_memory()
  policy = define_policy(model_name)

  if (model_name == "DQN"):
    model = DQNAgent(model=model_structure, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               enable_double_dqn=True, dueling_type='avg', target_model_update=1e-2)
  elif (model_name == "DDPG"):
    action_input, critic_layers = define_critic_layers(env)
    random_process = define_random_process(nb_actions)
    model = DDPGAgent(nb_actions=nb_actions, actor=model_structure, critic=critic_layers, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
    
  model.compile(Adam(lr=1e-3), metrics=['mae'])
  model.fit(env, nb_steps=50000, visualize=False, verbose=2)
  model.save_weights('model_from_MAIN_HELPER_{}_weights.h5f'.format(env_name), overwrite=True)
  model.test(env, nb_episodes=5, visualize=False)