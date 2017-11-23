import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory, Transition
from mlp_actor import MLP as MLPA
from mlp_critic import MLP as MLPC
from phase_mlp_actor import PMLP as PMLPA
from phase_mlp_critic import PMLP as PMLPC

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

class DDPG():
    """
    The Deep Deterministic Policy Gradient (DDPG) Agent
    Parameters
    ----------
        actor_optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate and other
            parameters for the optimizer
        critic_optimizer_spec: OptimizerSpec
        num_feature: int
            The number of features of the environmental state
        num_action: int
            The number of available actions that agent can choose from
        replay_memory_size: int
            How many memories to store in the replay memory.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        tau: float
            The update rate that target networks slowly track the learned networks.
    """
    def __init__(self,
                 actor_optimizer_spec,
                 critic_optimizer_spec,
                 num_feature,
                 num_action,
                 net_type,
                 replay_memory_size=1000000,
                 batch_size=64,
                 tau=0.001):
        ###############
        # BUILD MODEL #
        ###############
        self.num_feature = num_feature
        self.num_action = num_action
        self.batch_size = batch_size
        self.tau = tau
        # Construct actor and critic

        if net_type == 0:
                self.actor = MLPA(input_size=num_feature, output_size=num_action, hidden_size=(400,300), n_layers=2, tanh_flag=1).type(dtype)
                self.target_actor = MLPA(input_size=num_feature, output_size=num_action, hidden_size=(400,300), n_layers=2, tanh_flag=1).type(dtype)
                self.critic = MLPC(input_size_state=num_feature, input_size_action=num_action, output_size=1, hidden_size=(400,300), n_layers=2).type(dtype)
                self.target_critic = MLPC(input_size_state=num_feature, input_size_action=num_action, output_size=1, hidden_size=(400,300), n_layers=2).type(dtype)
        elif net_type == 1:
                self.actor = MLPA(input_size=num_feature+1, output_size=num_action, hidden_size=(400,300), n_layers=2, tanh_flag=1).type(dtype)
                self.target_actor = MLPA(input_size=num_feature+1, output_size=num_action, hidden_size=(400,300), n_layers=2, tanh_flag=1).type(dtype)
                self.critic = MLPC(input_size_state=num_feature+1, input_size_action=num_action, output_size=1, hidden_size=(400,300), n_layers=2).type(dtype)
                self.target_critic = MLPC(input_size_state=num_feature+1, input_size_action=num_action, output_size=1, hidden_size=(400,300), n_layers=2).type(dtype)
        elif net_type == 2:
                self.actor = PMLPA(input_size=num_feature, output_size=num_action, hidden_size=(400,300), dtype=dtype, n_layers=2, tanh_flag=1).type(dtype)
                self.target_actor = PMLPA(input_size=num_feature, output_size=num_action, hidden_size=(400,300), dtype=dtype, n_layers=2, tanh_flag=1).type(dtype)
                self.critic = PMLPC(input_size_state=num_feature, input_size_action=num_action, output_size=1, hidden_size=(400,300), dtype=dtype, n_layers=2).type(dtype)
                self.target_critic = PMLPC(input_size_state=num_feature, input_size_action=num_action, output_size=1, hidden_size=(400,300), dtype=dtype, n_layers=2).type(dtype)


        # Construct the optimizers for actor and critic
        self.actor_optimizer = actor_optimizer_spec.constructor(self.actor.parameters(), **actor_optimizer_spec.kwargs)
        self.critic_optimizer = critic_optimizer_spec.constructor(self.critic.parameters(), **critic_optimizer_spec.kwargs)
        # Construct the replay memory
        self.replay_memory = ReplayMemory(replay_memory_size)


    def copy_weights_for_finetune(self, weight_files):
        # hard coded for finetuning ...

		# copy actor
	for lin_layer, weight_file in zip(self.actor.control_hidden_list[0], weight_files):
        	agent = torch.load(weight_file)
	        lin_layer.load_state_dict(agent.actor.l1.state_dict())

	for lin_layer, weight_file in zip(self.actor.control_hidden_list[1], weight_files):
		agent = torch.load(weight_file)
	        lin_layer.load_state_dict(agent.actor.l2.state_dict())

	for lin_layer, weight_file in zip(self.actor.control_h2o_list, weight_files):
		agent = torch.load(weight_file)
	        lin_layer.load_state_dict(agent.actor.h2o.state_dict())


	# copy critic
	for lin_layer, weight_file in zip(self.critic.control_hidden_list[0], weight_files):
	        agent = torch.load(weight_file)
	        lin_layer.load_state_dict(agent.critic.l1.state_dict())

	for lin_layer, weight_file in zip(self.critic.control_hidden_list[1], weight_files):
        	agent = torch.load(weight_file)
	        lin_layer.load_state_dict(agent.critic.l2.state_dict())

	for lin_layer, weight_files in zip(self.critic.control_h2o_list, weight_files):
	        agent = torch.load(weight_file)
	        lin_layer.load_state_dict(agent.critic.h2o.state_dict())

    def select_action(self, state, phase, net_type):
        state = torch.from_numpy(state).type(dtype).unsqueeze(0)
        phase = torch.from_numpy(np.array([phase])).type(dtype).unsqueeze(0)
        if net_type == 0:
                action = self.actor(Variable(state, volatile=True)).data.cpu()
        elif net_type == 1:
                action = self.actor(Variable(torch.cat((state,phase),1), volatile=True)).data.cpu()
        elif net_type == 2:
                action = self.actor(Variable(state, volatile=True), Variable(phase, volatile=True)).data.cpu()
                #self.actor.reset()

        return action

    def update(self, net_type, gamma=1.0):
        if len(self.replay_memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, phase_batch, next_phase_batch, done_mask = \
            self.replay_memory.sample(self.batch_size)
        state_batch = Variable(torch.from_numpy(state_batch).type(dtype))
        action_batch = Variable(torch.from_numpy(action_batch).type(dtype))
        reward_batch = Variable(torch.from_numpy(reward_batch).type(dtype))
        next_state_batch = Variable(torch.from_numpy(next_state_batch).type(dtype))
        phase_batch = Variable(torch.from_numpy(phase_batch).type(dtype)).unsqueeze(1)
        next_phase_batch = Variable(torch.from_numpy(next_phase_batch).type(dtype)).unsqueeze(1)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

        ### Critic ###
        self.critic_optimizer.zero_grad()
        if net_type == 0 or net_type == 1:
            if net_type == 0:
                # Compute current Q value, critic takes state and action choosen
                current_Q_values = self.critic(torch.cat((state_batch, action_batch),1))
                # Compute next Q value based on which action target actor would choose
                # Detach variable from the current graph since we don't want gradients for next Q to propagated
                #target_actions = self.target_actor(state_batch) # shouldn't it be next_state_batch
                target_actions = self.target_actor(next_state_batch)
                next_max_q = self.target_critic(torch.cat((next_state_batch, target_actions),1)).detach().max(1)[0]
            elif net_type == 1:
                # Compute current Q value, critic takes state and action choosen
                current_Q_values = self.critic(torch.cat((state_batch, phase_batch, action_batch),1))
                # Compute next Q value based on which action target actor would choose
                # Detach variable from the current graph since we don't want gradients for next Q to propagated
                #target_actions = self.target_actor(torch.cat((state_batch, phase_batch),1)) shouldn't it be next_state_batch, next_phase_batch
                target_actions = self.target_actor(torch.cat((next_state_batch, next_phase_batch),1))
                next_max_q = self.target_critic(torch.cat((next_state_batch, next_phase_batch, target_actions),1)).detach().max(1)[0]

            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = reward_batch + (gamma * next_Q_values)
            # Compute Bellman error (using Huber loss)
            #critic_loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
	    critic_loss = F.mse_loss(current_Q_values, target_Q_values)
            # Optimize the critic
            critic_loss.backward()
            self.critic_optimizer.step()

        elif net_type == 2:
            current_Q_values = self.critic(torch.cat((state_batch, action_batch),1), phase_batch)
            target_actions = self.target_actor(next_state_batch, next_phase_batch)
            next_max_q = self.target_critic(torch.cat((next_state_batch, target_actions),1), next_phase_batch).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            target_Q_values = reward_batch + (gamma * next_Q_values)
            critic_loss = F.mse_loss(current_Q_values, target_Q_values)
            critic_loss.backward()
            # Optimize the critic
            self.critic_optimizer.step()


        ### Actor ###
        self.actor_optimizer.zero_grad()
        if net_type == 0 or net_type == 1:
            if net_type == 0:
                actor_loss = -self.critic(torch.cat((state_batch, self.actor(state_batch)),1)).mean()
            elif net_type == 1:
                actor_loss = -self.critic(torch.cat((state_batch, phase_batch, self.actor(torch.cat((state_batch, phase_batch),1))),1)).mean()

            # Optimize the actor
            actor_loss.backward()
            self.actor_optimizer.step()


        elif net_type == 2:
            actor_loss = -self.critic(torch.cat((state_batch, self.actor(state_batch, phase_batch)),1), phase_batch).mean()
            actor_loss.backward()

            # Optimize the actor
            self.actor_optimizer.step()

        # Update the target networks
        self.update_target(self.target_critic, self.critic)
        self.update_target(self.target_actor, self.actor)

    def update_target(self, target_model, model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
