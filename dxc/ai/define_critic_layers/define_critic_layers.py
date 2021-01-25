def define_critic_layers(env):
  action_input = Input(shape=(env.action_space.shape[0],), name='action_input')
  observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
  flattened_observation = Flatten()(observation_input)
  x = Concatenate()([action_input, flattened_observation])
  x = Dense(16)(x) # was 32
  x = Activation('relu')(x)
  x = Dense(16)(x)
  x = Activation('relu')(x)
  x = Dense(16)(x)
  x = Activation('relu')(x)
  x = Dense(1)(x)
  x = Activation('linear')(x)
  critic = Model(inputs=[action_input, observation_input], outputs=x)

  return action_input, critic