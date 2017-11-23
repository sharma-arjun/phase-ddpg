import sys
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from itertools import product
import gym
from gym import wrappers
from utils.normalized_env import NormalizedEnv
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Phase():
	def __init__(self):
		self.n = 8
		self.l = np.linspace(1,1.5,self.n/2) # hopper
		#self.l = np.linspace(0.8,2.0,(self.n+2)/2) # walker


	# hopper
    	def comp_phase(self, height, vel):
        	if height <= 1.0:
	        	phase = 0
        	elif height > 1.5:
	        	phase = math.pi
        	else:
	        	for i in range(self.n/2-1):
        	        	if height > self.l[i] and height <= self.l[i+1]:
			                phase = (2*math.pi/self.n)*(i+1)
        	if vel < 0:
		        phase = 2*math.pi - phase

	        return phase

	# walker
	#def comp_phase(self, height, vel):
        #	if height <= 0.8:
        #		phase = 0
	#        elif height > 2.0:
        #		phase = math.pi
	#        else:
        #    		for i in range(self.n/2):
        #        		if height > self.l[i] and height <= self.l[i+1]:
        #            			phase = (2*math.pi/self.n)*(i)
        #	if vel < 0:
	#		phase = 2*math.pi - phase

	#        return phase


	def reset(self):
		self.timer = 0

def main():

	phase_obj = Phase()
	net_type = int(sys.argv[1])
	agent = torch.load(sys.argv[2])
        actor = agent.actor
	env = NormalizedEnv(gym.make('Hopper-v1'))
	avg_total_reward = 0
	avg_step_count = 0

	print 'Using greedy policy ...'
	for ep in range(100):
		actor.reset()
		s = env.reset()
		phase_obj.reset()
		total_reward = 0
		step_count = 0
		terminal = False
		while terminal == False:

	        	phase = phase_obj.comp_phase(env.env.env.model.data.qpos[1,0], env.env.env.model.data.qvel[1,0])
			if net_type == 0:
				x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a).data.cpu().numpy()
			elif net_type == 1:
				inp = np.concatenate((s,np.asarray([phase])))
				x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a).data.cpu().numpy()
			elif net_type == 2:
				x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a, Variable(torch.from_numpy(np.asarray([[phase]])))).data.cpu().numpy()



        	        s_prime, reward, terminal, info = env.step(a)
			total_reward += reward
			step_count += 1

			s = s_prime

		avg_total_reward += total_reward
		avg_step_count += step_count


	print 'Average reward ', avg_total_reward/100.0
	print 'Average step count ', avg_step_count/100.0

if __name__ == '__main__':
	main()
