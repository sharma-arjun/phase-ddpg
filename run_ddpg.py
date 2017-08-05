from collections import namedtuple
import sys
import gym
#import phase_envs.envs.AntEnv2 as AntEnv2
import torch.optim as optim

from utils.random_process import OrnsteinUhlenbeckProcess
from utils.normalized_env import NormalizedEnv
from agents.ddpg_low_dim import DDPG
from ddpg_learning import ddpg_learning

NUM_EPISODES = 10000
BATCH_SIZE = 64
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
Q_WEIGHT_DECAY = 0.01
TAU = 0.001
THETA = 0.15
SIGMA = 0.2
LOG_EVERY_N_EPS = 10
SAVE_EVERY_N_EPS = 500
MAX_EP_LENGTH = 200
NET_TYPE = int(sys.argv[1])
CHECKPOINT_NAME = sys.argv[2]

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

actor_optimizer_spec = OptimizerSpec(
    constructor=optim.Adam,
    kwargs=dict(lr=ACTOR_LEARNING_RATE),
)

critic_optimizer_spec = OptimizerSpec(
    constructor=optim.Adam,
    kwargs=dict(lr=CRITIC_LEARNING_RATE, weight_decay=Q_WEIGHT_DECAY),
)

env = NormalizedEnv(gym.make('Pendulum-v0'))
#env = NormalizedEnv(gym.make('Ant-v2'))

random_process = OrnsteinUhlenbeckProcess(theta=THETA, sigma=SIGMA, size=env.action_space.shape[0], sigma_min=0.05, n_steps_annealing=1000000)

agent = DDPG(
    actor_optimizer_spec=actor_optimizer_spec,
    critic_optimizer_spec=critic_optimizer_spec,
    num_feature=env.observation_space.shape[0],
    num_action=env.action_space.shape[0],
    net_type=NET_TYPE,
    replay_memory_size=REPLAY_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    tau=TAU
)

stats = ddpg_learning(
    env=env,
    random_process=random_process,
    agent=agent,
    net_type=NET_TYPE,
    num_episodes=NUM_EPISODES,
    checkpoint_name=CHECKPOINT_NAME,
    gamma=GAMMA,
    log_every_n_eps=LOG_EVERY_N_EPS,
    save_every_n_eps=SAVE_EVERY_N_EPS,
    max_ep_length=MAX_EP_LENGTH
)
