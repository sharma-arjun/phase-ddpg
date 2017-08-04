import numpy as np
from collections import defaultdict
from itertools import count
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory
from utils import plotting

class Phase():
    def __init__(self):
        self.phase_list = [0, math.pi/2, math.pi, 3*math.pi/2]
        self.timer = 0

    def comp_phase(self):

        if self.timer == 0:
            self.phase = random.choice(self.phase_list)
            self.timer += 1
        elif self.timer == 30:
            self.phase = random.choice(self.phase_list)
            self.timer = 1
        else:
            self.timer +=1

        return self.phase

    def reset(self):
        self.timer = 0


def ddpg_learning(
    env,
    random_process,
    agent,
    net_type,
    num_episodes,
    checkpoint_name,
    gamma=0.99,
    log_every_n_eps=10,
    save_every_n_eps=500
    ):

    """The Deep Deterministic Policy Gradient algorithm.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    random_process: Defined in utils.random_process
        The process that add noise for exploration in deterministic policy.
    agent:
        a DDPG agent consists of a actor and critic.
    net_type:
        MLP, MLP with phase input, Phase MLP architecture
    num_episodes:
        Number of episodes to run for.
    gamma: float
        Discount Factor
    log_every_n_eps: int
        Log and plot training info every n episodes.
    """
    ###############
    # RUN ENV     #
    ###############
    stats = plotting.EpisodeStats(
        episode_lengths=[],
        episode_rewards=[],
        mean_rewards=[])
    total_timestep = 0

    phase_obj = Phase()
    print 'Writing to plotfiles/' + checkpoint_name + '.txt'
    f = open('plotfiles/' + checkpoint_name + '.txt', 'w')

    for i_episode in range(num_episodes):
        #print 'Episode', i_episode
        state = env.reset()
        random_process.reset_states()
        phase_obj.reset()

        episode_reward = 0
        episode_length = 0

        for t in count(1):
            #print 't', t
            phase = phase_obj.comp_phase()
            env.env.env.phase = phase
            action = agent.select_action(state, phase, net_type).squeeze(0).numpy()
            # Add noise for exploration
            noise = random_process.sample()
            action += noise
            action = np.clip(action, -1.0, 1.0)
            next_state, reward, done, _ = env.step(action)
            # Update statistics
            total_timestep += 1
            episode_reward += reward
            episode_length = t
            # Store transition in replay memory
            agent.replay_memory.push(state, action, reward, next_state, phase, phase, done)
            # Update
            agent.update(net_type, gamma)
            if done:
                stats.episode_lengths.append(episode_length)
                stats.episode_rewards.append(episode_reward)
                mean_reward = np.mean(stats.episode_rewards[-100:])
                stats.mean_rewards.append(mean_reward)
                break
            else:
                state = next_state

        if i_episode % log_every_n_eps == 0:
            #pass
            print("### EPISODE %d ### TAKES %d TIMESTEPS" % (i_episode + 1, stats.episode_lengths[i_episode]))
            print("MEAN REWARD (100 episodes): " + "%.3f" % (mean_reward))
            print("TOTAL TIMESTEPS SO FAR: %d" % (total_timestep))

            f.write(str(mean_reward) + ' ' + str(total_timestep) + '\n')
            #plotting.plot_episode_stats(stats)

        if (i_episode + 1) % save_every_n_eps == 0:
            f_w = open('checkpoints/' + checkpoint_name + '_' + str(i_episode+1) + '_' + str(mean_reward)  + '.pth','wb')
	    torch.save(agent,f_w)

    f.close()

    return stats
