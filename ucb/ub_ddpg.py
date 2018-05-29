from __future__ import division
from collections import deque
import os
import warnings

import numpy as np
import keras.backend as K
import keras.layers as layers
import keras.optimizers as optimizers

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, nb_players, critic_action_inputs, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        assert len(critic_action_inputs) == nb_players
        if hasattr(actor.output, '__len__') and len(actor.output) != nb_players:
            raise ValueError((
                'Actor "{}" does not have the right number of ',
                'outputs. DDPG expects an actor that has {} outputs.'
                ).format(actor, nb_players))
        # if hasattr(critic.output, '__len__') and len(critic.output) > 1:
        #     raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        for critic_action_input in critic_action_inputs:
            if critic_action_input not in critic.input:
                raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at least two inputs, one for the action and one for the observation.'.format(critic))

        super(DDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.nb_players = nb_players
        self.critic_action_inputs = critic_action_inputs
        self.critic_action_input_idxes = [
            self.critic.input.index(critic_action_input)
            for critic_action_input in critic_action_inputs
        ]
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            y_true = K.squeeze(y_true, axis=-1)
            y_pred = K.squeeze(y_pred, axis=-1)
            loss = K.mean(
                # K.random_uniform(shape=(self.batch_size, self.nb_players), minval=0., maxval=1.) *
                huber_loss(y_true, y_pred, self.delta_clip),
                axis=-1)
            # y_true = K.print_tensor(y_true, message='y_true: ')
            # y_pred = K.print_tensor(y_pred, message='y_pred: ')
            # loss = K.print_tensor(loss, message='loss: ')
            return loss

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(
            optimizer=critic_optimizer,
            loss=[clipped_error]*self.nb_players,
            metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        critic_inputs = []
        actor_inputs = []
        for i in self.critic.input:
            if i in self.critic_action_inputs:
                critic_inputs.append([])
            else:
                critic_inputs.append(i)
                actor_inputs.append(i)
        actor_outputs = self.actor(actor_inputs)
        if not isinstance(actor_outputs, (list,)):
            actor_outputs = [actor_outputs]
        assert len(actor_outputs) == self.nb_players
        for input_idx, actor_output in zip(self.critic_action_input_idxes, actor_outputs):
            critic_inputs[input_idx] = actor_output

        # critic_outputs = layers.Maximum()(self.critic(critic_inputs))
        critic_outputs = self.critic(critic_inputs)
        if not isinstance(critic_outputs, (list,)):
            critic_outputs = [critic_outputs]
        assert len(critic_outputs) == self.nb_players

        actor_losses = [None]* self.nb_players
        for input_idx, critic_output in zip(self.critic_action_input_idxes, critic_outputs):
            actor_losses[input_idx] = -K.mean(critic_output)
        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights,
            loss=actor_losses)
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(actor_inputs + [K.learning_phase()],
                                             actor_outputs, updates=updates)
        else:
            if self.uses_learning_phase:
                actor_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(actor_inputs, actor_outputs, updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        # actions = [action.flatten() for action in self.actor.predict_on_batch(batch)]
        actions =  self.actor.predict_on_batch(batch)
        if self.nb_players == 1:
            actions =[actions]
        # actions = [a.flatten() for a in actions]
        assert len(actions) == self.nb_players
        # assert actions[0].shape == (self.nb_actions,)
        assert actions[0].shape == (1, self.nb_actions)
        # print('actions: {}'.format(actions))

        if len(self.critic.inputs) > (self.nb_players+1): # state is a list
            state_batch_with_action = batch[:]
        else:
            state_batch_with_action = [batch]
        for action_idx, input_idx in enumerate(self.critic_action_input_idxes):
            state_batch_with_action.insert(input_idx, actions[action_idx])
        q_values = [
            qv.flatten() 
            for qv in self.critic.predict_on_batch(state_batch_with_action)
        ]
        assert q_values[0].shape == (1, )
        assert len(q_values) == self.nb_players
        # print('q_values: {}'.format(q_values))

        action_best = actions[np.argmax(q_values)].flatten()
        # assert action_best.shape == (self.nb_actions, )
        assert action_best.shape == (self.nb_actions, )
        # print('action_best: {}'.format(action_best))
        # print(type(action_best[0]))

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action_best.shape
            action_best += noise

        return action_best

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                if not isinstance(target_actions, (list,)):
                    target_actions = [target_actions]
                assert len(target_actions) == self.nb_players
                assert target_actions[0].shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) > (self.nb_players+1): # state is a list
                # if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                # state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                for action_idx, input_idx in enumerate(self.critic_action_input_idxes):
                    state1_batch_with_action.insert(input_idx, target_actions[action_idx])
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action)
                if not isinstance(target_q_values, (list,)):
                    target_q_values = [target_q_values]
                target_q_values = [ tqv.flatten() for tqv in target_q_values]
                assert target_q_values[0].shape == reward_batch.shape
                assert len(target_q_values) == self.nb_players

                # Compute r_t + gamma * Q(s_t+1, mu(s_t+1)) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = [
                    self.gamma * terminal1_batch * tqv
                    for tqv in target_q_values
                ]
                assert discounted_reward_batch[0].shape == reward_batch.shape
                targets = [reward_batch + drb for drb in discounted_reward_batch] # .reshape(self.batch_size, 1)
                assert targets[0].shape == reward_batch.shape
                assert len(targets) == self.nb_players

                # Perform a single batch update on the critic network.
                # if len(self.critic.inputs) >= 3:
                if len(self.critic.inputs) > (self.nb_players+1): # state is a list
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                for input_idx in self.critic_action_input_idxes:
                    state0_batch_with_action.insert(input_idx, action_batch)
                # state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(
                    state0_batch_with_action,
                    targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

                # q_values = self.critic.predict_on_batch(state0_batch_with_action)
                # if not isinstance(q_values, (list,)):
                #     q_values = [q_values]
                # q_values = [ qv.flatten() for qv in q_values]
                # print('gamma: {}'.format(self.gamma))
                # print('terminal1_batch: {}'.format(terminal1_batch))
                # print('target_q_values: {}'.format(target_q_values))
                # print('discounted_reward_batch: {}'.format(discounted_reward_batch))
                # print('reward_batch: {}'.format(reward_batch))
                # print('targets: {}'.format(targets))
                # print('current q values: {}'.format(q_values))


            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)
                assert len(action_values) == self.nb_players
                assert action_values[0].shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics
