import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


ENV_NAME = 'MountainCarContinuous-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

nb_players = 8
# Next, we build a very simple model.

observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)

# Actor
x = Dense(16)(flattened_observation)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
a_heads = []
# create multiple heads each independently computing the Q value
for pidx in range(nb_players):
    y = Dense(16)(x)
    y = Activation('relu')(y)
    y = Dense(16)(y)
    y = Activation('relu')(y)
    y = Dense(nb_actions)(y)
    y = Activation('linear')(y)
    a_heads.append(y)
actor = Model(inputs=[observation_input], outputs=a_heads)
print(actor.summary())

#Critic
action_inputs = [
    Input(shape=(nb_actions,), name='action_input_{}'.format(pidx))
    for pidx in range(nb_players)
]
x = Dense(32)(flattened_observation)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
# create multiple heads each independently computing the Q value
q_heads = []
for idx in range(nb_players):
    y = Concatenate()([action_inputs[idx], x])
    y = Dense(32)(y)
    y = Activation('relu')(y)
    y = Dense(16)(y)
    y = Activation('relu')(y)
    y = Dense(1)(y)
    y = Activation('linear')(y)
    q_heads.append(y)
critic = Model(inputs=action_inputs + [observation_input], outputs=q_heads)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
agent = DDPGAgent(
    nb_actions=nb_actions, actor=actor, critic=critic,
    nb_players=nb_players, critic_action_inputs=action_inputs,
    memory=memory, nb_steps_warmup_critic=400, nb_steps_warmup_actor=400,
    random_process=None, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=1000000, visualize=True, verbose=1, nb_max_episode_steps=1000)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=800)
