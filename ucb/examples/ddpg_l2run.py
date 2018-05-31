import numpy as np
import gym
import types
import time
import pathlib
import logging


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, concatenate
from keras.optimizers import Adam

from ucb.ub_ddpg import UBDDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from osim.env import L2RunEnv
from osim.env.utils.mygym import convert_to_gym


def gymify_osim_env(env):
    env.action_space = ( [-1.0] * env.osim_model.get_action_space_size(), [1.0] * env.osim_model.get_action_space_size() )
    env.action_space = convert_to_gym(env.action_space)

    env._step = env.step

    def step(self, action):
        return self._step(action * 2 - 1)

    env.step = types.MethodType(step, env)
    return env

# Get the environment and extract the number of actions.
env = L2RunEnv(visualize=False)
# env = gymify_osim_env(L2RunEnv(visualize=False))
env.reset()

np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

nb_players = 16
# Next, we build a very simple model.

observation_input = Input(
    shape=(1,) + env.observation_space.shape,
    name='actor_observation_input')
flattened_observation = Flatten()(observation_input)

# Actor
x = Dense(64)(flattened_observation)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
a_heads = []
# create multiple heads each independently computing the Q value
for pidx in range(nb_players):
    y = Dense(32)(x)
    y = Activation('relu')(y)
    y = Dense(32)(y)
    y = Activation('relu')(y)
    y = Dense(nb_actions)(y)
    y = Activation('sigmoid')(y)
    a_heads.append(y)
actor = Model(inputs=[observation_input], outputs=a_heads)
print(actor.summary())

#Critic
action_inputs = [
    Input(shape=(nb_actions,), name='action_input_{}'.format(pidx))
    for pidx in range(nb_players)
]

observation_input = Input(
    shape=(1,) + env.observation_space.shape,
    name='critic_observation_input')
flattened_observation = Flatten()(observation_input)

x = Dense(64)(flattened_observation)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
# create multiple heads each independently computing the Q value
q_heads = []
for idx in range(nb_players):
    y = Concatenate()([action_inputs[idx], x])
    y = Dense(64)(y)
    y = Activation('relu')(y)
    y = Dense(32)(y)
    y = Activation('relu')(y)
    y = Dense(1)(y)
    y = Activation('linear')(y)
    q_heads.append(y)
critic = Model(inputs=action_inputs + [observation_input], outputs=q_heads)
print(critic.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
agent = UBDDPGAgent(
    nb_actions=nb_actions, actor=actor, critic=critic,
    nb_players=nb_players, critic_action_inputs=action_inputs,
    memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
    random_process=None, gamma=.99, target_model_update=1e-3, delta_clip=1.)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


env_name = 'l2run'
current_time_sec = int(time.time())
base_dir = 'log/l2run/{}'.format(current_time_sec)
pathlib.Path(base_dir).mkdir(parents=True, exist_ok=False)
logging.basicConfig(filename='{}/{}.log'.format(base_dir, env_name), level=logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)
logging.info('setting base directory to {}'.format(base_dir))

checkpoint_weights_filename = '{}/ddpg_{}_weights_{{step}}.h5f'.format(base_dir, env_name)
filelogger_file = '{}/ddpg_{}_log.json'.format(base_dir, env_name)
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
callbacks += [FileLogger(filelogger_file, interval=1000)]

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = agent.fit(env, nb_steps=1000000, visualize=False, verbose=1,
        nb_max_episode_steps=1000, callbacks=callbacks)
training_history_file = '{}/ddpg_{}_training_history.pickle'.format(base_dir, env_name)
with open(training_history_file, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# After training is done, we save the final weights.
agent.save_weights('ddpg_l2run_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=800)
