import sys
import copy
import math
import random
import numpy as np
from phase_lstm_multilayer_new import PLSTM
from lstm import LSTM
#from mlp import MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from itertools import product
import gym
from gym import wrappers
#import phase_envs.envs.AntEnv2 as AntEnv2


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def create_targets(memory, q_vals, target_critic, target_actor, net_type, gamma=1):
	# memory: 0 - current_state 1: action (numpy) 2: reward 3: next state 4: current phase 5: next phase 6: terminal
	n_eps = len(memory)
	q_target = torch.zeros((n_eps, 1))
	
	for i in range(n_eps):
		phase_prime = memory[i][5]
		if net_type == 0:
			s_prime = torch.from_numpy(np.array(memory[i][3])).type(dtype)
			x_a_prime = Variable(s_prime, requires_grad=False).unsqueeze(0)
			a_prime = target_actor.forward(x_a_prime)
			x_c_prime = Variable(torch.cat((a_prime.data.squeeze(0),s_prime),0), requires_grad=False).unsqueeze(0)
			q_target[i,:] = memory[i][2] + gamma*target_critic.forward(x_c_prime).data*(1-float(memory[i][6]))
		elif net_type == 1:
			inp = np.concatenate((np.array(memory[i][3]), np.asarray([phase_prime])))
			s_prime = torch.from_numpy(inp).type(dtype)
			x_a_prime = Variable(s_prime, requires_grad=False).unsqueeze(0)
			a_prime = target_actor.forward(x_a_prime)
			x_c_prime = Variable(torch.cat((a_prime.data.squeeze(0),s_prime),0), requires_grad=False).unsqueeze(0)
			q_target[i,:] = memory[i][2] + gamma*target_critic.forward(x_c_prime).data*(1-float(memory[i][6]))
		elif net_type == 2:
			s_prime = torch.from_numpy(np.array(memory[i][3])).type(dtype)
			x_a_prime = Variable(s_prime, requires_grad=False).unsqueeze(0)
			a_prime = target_actor.forward(x_a_prime, phase_prime)
			x_c_prime = Variable(torch.cat((a_prime.data.squeeze(0),s_prime),0), requires_grad=False).unsqueeze(0)
			q_target[i,:] = memory[i][2] + gamma*target_critic.forward(x_c_prime, phase_prime).data*(1-float(memory[i][6]))
		#elif net_type == 3:
		#	s_prime = torch.from_numpy(np.array(memory[i][3])).type(dtype)
		#	s_prime = Variable(s_prime, requires_grad=False).unsqueeze(0)
		#	a_prime = target_actor.forward(x_a_prime)
		#	x_c_prime = Variable(torch.cat((a_prime.data.squeeze(0),s_prime),0), requires_grad=False).unsqueeze(0)
		#	q_target[i,:] = memory[i][2] + gamma*target_critic.forward(x_c_prime).data*(1-float(memory[i][6]))
		#elif net_type == 4:
		#	inp = np.concatenate((np.array(memory[i][3]), np.asarray([phase_prime])))
		#	s_prime = torch.from_numpy(inp).type(dtype)
		#	x_a_prime = Variable(s_prime, requires_grad=False).unsqueeze(0)
		#	a_prime = target_actor.forward(x_a_prime)
		#	x_c_prime = Variable(torch.cat((a_prime.data.squeeze(0),s_prime),0), requires_grad=False).unsqueeze(0)
		#	q_target[i,:] = memory[i][2] + gamma*target_critic.forward(x_c_prime).data*(1-float(memory[i][6]))

	target_actor.reset()
	target_critic.reset()
	return q_target

class ExperienceReplay():
	def __init__(self, max_memory_size = 100):
		self.memory = []
		self.oldest = -1
		self.max_memory_size = max_memory_size
	
	def add(self, experience):
		if len(self.memory) < self.max_memory_size: 
			self.memory.append(experience)
			self.oldest = 0
		else:
			self.memory.insert(self.oldest, experience)
			self.oldest = (self.oldest + 1) % self.max_memory_size

	def sample(self):
		idx = np.random.randint(0, high=len(self.memory))
		return self.memory[idx]


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


class OrnsteinUhlenbeck():
	def __init__(self, theta, sigma, sigma_min, mu, n, dt=1):
		self.theta = theta
		self.sigma = sigma
		self.sigma_min = sigma_min
		self.mu = mu
		#self.x = sigma*np.random.multivariate_normal(mean=np.zeros((n,)), cov=np.eye(n))
		self.x = np.zeros((n,))
		self.dt = dt
		self.n = n

	def sample(self, ep_n, N):
		sigma = max(self.sigma - ((self.sigma - self.sigma_min)/N)*ep_n, self.sigma_min)
		self.x_prime = self.x + self.theta*(self.mu-self.x)*self.dt + sigma*math.sqrt(self.dt)*np.random.multivariate_normal(mean=np.zeros((self.n,)), cov=np.eye(self.n))
		self.x = self.x_prime
		return self.x_prime

	def reset(self):
		#self.x = np.random.multivariate_normal(mean=np.zeros((self.n,)), cov=self.sigma*np.eye(self.n))
		self.x = np.zeros((self.n,))

def main():
	max_episode_length = 200
	n_episodes = 100000
	n_save_every = 1000
	burn_in = 100
	net_type = int(sys.argv[1])

	env = gym.make('Ant-v2')
	M = ExperienceReplay(max_memory_size=1000)
	s = env.reset()
	env_output_size = env.action_space.shape[0]
	noise = OrnsteinUhlenbeck(theta=0.15, sigma=0.2, sigma_min=0.01, mu=np.zeros((env_output_size,)), n=env_output_size)
	phase_obj = Phase()
	tau = 0.001


	if net_type == 0: # rnn without phase
		actor = LSTM(input_size=s.shape[0], output_size=env_output_size, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, scale=float(env.action_space.high[0]), tanh_flag=1).type(dtype)
		critic = LSTM(input_size=s.shape[0]+env_output_size, output_size=1, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, tanh_flag=0).type(dtype)
		target_actor = LSTM(input_size=s.shape[0], output_size=env_output_size, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, scale=float(env.action_space.high[0]), tanh_flag=1).type(dtype)
		target_critic = LSTM(input_size=s.shape[0]+env_output_size, output_size=1, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, tanh_flag=0).type(dtype)
	elif net_type == 1: # rnn with phase as additional input
		actor = LSTM(input_size=s.shape[0]+1, output_size=env_output_size, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, scale=float(env.action_space.high[0]), tanh_flag=1).type(dtype)
		critic = LSTM(input_size=s.shape[0]+env_output_size+1, output_size=1, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, tanh_flag=0).type(dtype)
		target_actor = LSTM(input_size=s.shape[0]+1, output_size=env_output_size, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, scale=float(env.action_space.high[0]), tanh_flag=1).type(dtype)
		target_critic = LSTM(input_size=s.shape[0]+env_output_size+1, output_size=1, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, tanh_flag=0).type(dtype)
	elif net_type == 2: # phase rnn
		actor = PLSTM(input_size=s.shape[0], output_size=env_output_size, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, scale=float(env.action_space.high[0]), tanh_flag=1).type(dtype)
		critic = PLSTM(input_size=s.shape[0]+env_output_size, output_size=1, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, tanh_flag=0).type(dtype)
		target_actor = PLSTM(input_size=s.shape[0], output_size=env_output_size, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, scale=float(env.action_space.high[0]), tanh_flag=1).type(dtype)
		target_critic = PLSTM(input_size=s.shape[0]+env_output_size, output_size=1, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, tanh_flag=0).type(dtype)
	#elif net_type == 3: # mlp without phase
	#	actor = MLP(input_size=s.shape[0], output_size=env_output_size, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, scale=float(env.action_space.high[0]), tanh_flag=1).type(dtype)
	#	critic = MLP(input_size=s.shape[0]+env_output_size, output_size=1, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, tanh_flag=0).type(dtype)
	#elif net_type == 4: # mlp with phase as additional input
	#	actor = MLP(input_size=s.shape[0]+1, output_size=env_output_size, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, scale=float(env.action_space.high[0]), tanh_flag=1).type(dtype)
	#	critic = MLP(input_size=s.shape[0]+env_output_size+1, output_size=1, hidden_size=300, dtype=dtype, n_layers=2, batch_size=1, tanh_flag=0).type(dtype)

	actor.reset()
	critic.reset()

	#target_actor = copy.deepcopy(actor)
	#target_critic = copy.deepcopy(critic)


	for p, p_t in zip(critic.parameters(), target_critic.parameters()):
		p_t.data.copy_(p.data)

	for p, p_t in zip(actor.parameters(), target_actor.parameters()):
		p_t.data.copy_(p.data)

	criterion = nn.SmoothL1Loss()
	criterion = nn.MSELoss()
	optimizer_critic = optim.Adam(critic.parameters(), lr=0.001, weight_decay=0.01)
	optimizer_actor = optim.Adam(actor.parameters(), lr=0.0001)

	list_of_total_rewards = []
	list_of_n_episodes = []

	#Burn in with random policy
	for i in range(burn_in):
		phase_obj.reset()
		phase = phase_obj.comp_phase()
		episode_experience = []
		for j in range(max_episode_length):
			#phase = phase_obj.comp_phase()
                        env.env.env.phase = phase
			a = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=env_output_size)
			s_prime, reward, terminal, info  = env.step(a)
			phase_prime = phase_obj.comp_phase()
			episode_experience.append((s,a,reward,s_prime,phase,phase_prime,terminal))
			if terminal == True:
				#print 'Reached goal state!'
				break
			s = s_prime
			phase = phase_prime

		M.add(episode_experience)
		s = env.reset()

	print 'Burn in completed'

	filename = 'plotfiles/' + sys.argv[2] + '.txt'
	print 'Writing to ' + filename
	f = open(filename,'w')

	for i in range(n_episodes):
		total_reward = 0
		episode_experience = []
		phase_obj.reset()
		phase = phase_obj.comp_phase()

		# zero gradients
		optimizer_actor.zero_grad()
		optimizer_critic.zero_grad()

		for j in range(max_episode_length):
			#phase = phase_obj.comp_phase()
                        env.env.env.phase = phase
			if net_type == 0:
				x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a)
				#x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=False).unsqueeze(0)
				#q = critic.forward(x_c)
			elif net_type == 1:
				inp = np.concatenate((s,np.asarray([phase])))
				x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a)
				#x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(inp).type(dtype)),0), requires_grad=False).unsqueeze(0)
				#q = critic.forward(x_c)
			elif net_type == 2:
				x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a, phase)
				#x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=False).unsqueeze(0)
				#q = critic.forward(x_c, phase)
			#elif net_type == 3:
			#	x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
			#	a = actor.forward(x_a)
			#	#x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=False).unsqueeze(0)
			#	#q = critic.forward(x_c)
			#elif net_type == 4:
			#	inp = np.concatenate((s.state,np.asarray([phase])))
			#	x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
			#	a = actor.forward(x_a)
			#	#x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(inp).type(dtype)),0), requires_grad=False).unsqueeze(0)
			#	#q = critic.forward(x_c)

			# action to be selected after addition of random noise (Ornstein-Uhlenbeck).
			a = np.clip(a.data.cpu().numpy() + noise.sample(i, n_episodes/2), env.action_space.low , env.action_space.high) # a is in numpy format now
			s_prime, reward, terminal, info = env.step(a)
                        s_prime = np.ndarray.flatten(s_prime)
			phase_prime = phase_obj.comp_phase()
			total_reward += reward
			episode_experience.append((s,a,reward,s_prime,phase,phase_prime,terminal))
			if terminal == True:
				#print 'Reached goal state!'
				break
			#q_vals.append(q)
			s = s_prime
			phase = phase_prime

		M.add(episode_experience)
		#print 'Episode lasted for %d steps.' % (j+1)
		#print 'Total reward collected: ', total_reward
		list_of_total_rewards.append(total_reward)
		list_of_n_episodes.append(j+1)
		if i % 100 == 0 and i > 0:
			print str(i) + ': Avg. Reward: ' + str(sum(list_of_total_rewards[i-100:i])/100.0) + ' Avg. Episode length: ' + str(sum(list_of_n_episodes[i-100:i])/100.0)

		# write to file for plotting
		f.write(str(total_reward) + ' ' + str(j+1) + '\n')

		actor.reset()
		#critic.reset()
		noise.reset()

		# save policy
		if i % n_save_every == 0 and i> 0:
			f_w = open('checkpoints/' + sys.argv[2] + '_' + str(i) + '_' + str(sum(list_of_total_rewards[i-100:i])/100.0)  + '.pth', 'wb')
			torch.save(actor,f_w)
			#torch.save(critic,f_w)

		# forward pass through memory sample
		memory = M.sample()
		q_vals = []
		x_c_vals = []
		a_vals = []
		for j in range(len(memory)):
			s = memory[j][0]
			phase = memory[j][4]
			if net_type == 0:
				#x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
				x_a = Variable(torch.from_numpy(s).type(dtype)).unsqueeze(0)
				a = actor.forward(x_a)
				#a = torch.from_numpy(memory[j][1])
				#x_c = torch.cat((a, s),1)
				#x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=True).unsqueeze(0)
				q = critic.forward(torch.cat((a,x_a),1))
			elif net_type == 1:
				inp = np.concatenate((s,np.asarray([phase])))
				x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a)
				#a = torch.from_numpy(memory[j][1])
				#x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(inp).type(dtype)),0), requires_grad=True).unsqueeze(0)
				q = critic.forward(torch.cat((a,x_a),1))
			elif net_type == 2:
				x_a = Variable(torch.from_numpy(s).type(dtype)).unsqueeze(0)
				a = actor.forward(x_a, phase)
				#a = torch.from_numpy(memory[j][1])
				#x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=True).unsqueeze(0)
				q = critic.forward(torch.cat((a,x_a),1), phase)
			#elif net_type == 3:
			#	x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
			#	a = actor.forward(x_a)
			#	#a = torch.from_numpy(memory[j][1])
			#	x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=True).unsqueeze(0)
			#	q = critic.forward(x_c)
			#elif net_type == 4:
			#	inp = np.concatenate((s.state,np.asarray([phase])))
			#	x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
			#	a = actor.forward(x_a)
			#	#a = torch.from_numpy(memory[j][1])
			#	x_c = Variable(torch.cat((a.data.squeeze(0),torch.from_numpy(inp).type(dtype)),0), requires_grad=True).unsqueeze(0)
			#	q = critic.forward(x_c)

			q_vals.append(q)
			a_vals.append(a)
			#x_c_vals.append(x_c)

		# backward pass for actor
		outputs_q = torch.stack(q_vals,0).squeeze(1)
		#grad_a = []
		#for idx_inp in range(len(x_c_vals)):
		#	grad_in = torch.autograd.grad(outputs=outputs_q[idx_inp:,:],inputs=x_c_vals[idx_inp],grad_outputs=torch.ones(len(q_vals)-idx_inp), retain_variables=True)[0].data
		#	grad_a.append(grad_in)

		#grad_a = torch.stack(grad_a, 0).squeeze(1)[:,0:env_output_size]
		#outputs_a = torch.stack(a_vals,0).squeeze(1)
		#loss = -outputs_a
		loss = -outputs_q.mean()
		loss.backward(retain_variables=False)

		# backward pass for critic
		critic.reset()
		optimizer_critic.zero_grad() # clear the gradients computed in the backward pass for actor
		targets_q = Variable(create_targets(memory, q_vals, target_critic, target_actor, net_type, gamma=0.99), requires_grad=False).type(dtype)

		q_vals = []	
		for j in range(len(memory)):
			s = memory[j][0]
			phase = memory[j][4]
			if net_type == 0:
				a = torch.from_numpy(memory[j][1]).type(dtype)
				x_c = Variable(torch.cat((a.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=True).unsqueeze(0)
				q = critic.forward(x_c)
			elif net_type == 1:
				inp = np.concatenate((s,np.asarray([phase])))
				a = torch.from_numpy(memory[j][1]).type(dtype)
				x_c = Variable(torch.cat((a.squeeze(0),torch.from_numpy(inp).type(dtype)),0), requires_grad=True).unsqueeze(0)
				q = critic.forward(x_c)
			elif net_type == 2:
				a = torch.from_numpy(memory[j][1]).type(dtype)
				x_c = Variable(torch.cat((a.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=True).unsqueeze(0)
				q = critic.forward(x_c, phase)
			#elif net_type == 3:
			#	a = torch.from_numpy(memory[j][1]).type(dtype)
			#	x_c = Variable(torch.cat((a.squeeze(0),torch.from_numpy(s).type(dtype)),0), requires_grad=True).unsqueeze(0)
			#	q = critic.forward(x_c)
			#elif net_type == 4:
			#	inp = np.concatenate((s.state,np.asarray([phase])))
			#	a = torch.from_numpy(memory[j][1]).type(dtype)
			#	x_c = Variable(torch.cat((a.squeeze(0),torch.from_numpy(inp).type(dtype)),0), requires_grad=True).unsqueeze(0)
			#	q = critic.forward(x_c)

			q_vals.append(q)

		predicted_q = torch.stack(q_vals,0).squeeze(1)
		loss = criterion(predicted_q, targets_q)
		loss.backward(retain_variables=False)

		# phase lstm step
		if net_type == 2:
			critic.update_control_gradients()
			actor.update_control_gradients()

		# clip gradients here ...
		nn.utils.clip_grad_norm(critic.parameters(), 5.0)
		for p in critic.parameters():
			p.data.add_(0.0001, p.grad.data)

		nn.utils.clip_grad_norm(actor.parameters(), 5.0)
		for p in actor.parameters():
			p.data.add_(0.0001, p.grad.data)

		# optimizer step
		optimizer_critic.step()
		optimizer_actor.step()

		# Reset environment and policy hidden vector at the end of episode
		actor.reset()
		critic.reset()
		s = env.reset()

		# soft copy into target network
		for p, p_t in zip(critic.parameters(), target_critic.parameters()):
			p_t.data.copy_(tau*p.data + (1-tau)*p_t.data)

		for p, p_t in zip(actor.parameters(), target_actor.parameters()):
			p_t.data.copy_(tau*p.data + (1-tau)*p_t.data)


		# hard copy into target networks
		#if i % n_copy_after == 0 and i > 0:
		#	target_actor = copy.deepcopy(actor)
		#	target_critic = copy.deepcopy(critic)

	# testing with greedy policy
	print 'Using greedy policy ...'
	s = env.reset()
	phase_obj.reset()
	total_reward = 0
	step_count = 0
	terminal = False
	while terminal == False:
		phase = phase_obj.comp_phase()
                env.env.env.phase = phase
		if net_type == 0:
			x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
			a = actor.forward(x_a).data.cpu().numpy()
			#x_c = Variable(torch.cat((a.data,torch.from_numpy(s)),1).type(dtype), requires_grad=False).unsqueeze(0)
			#q = critic.forward(x_c)
		elif net_type == 1:
			inp = np.concatenate((s,np.asarray([phase])))
			x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
			a = actor.forward(x_a).data.cpu().numpy()
			#x_c = Variable(torch.cat((a.data,torch.from_numpy(inp)),1).type(dtype), requires_grad=False).unsqueeze(0)
			#q = critic.forward(x_c)
		elif net_type == 2:
			x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
			a = actor.forward(x_a, phase).data.cpu().numpy()
			#x_c = Variable(torch.cat((a.data,torch.from_numpy(s)),1).type(dtype), requires_grad=False).unsqueeze(0)
			#q = critic.forward(x_c, phase)
		#elif net_type == 3:
		#	x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
		#	a = actor.forward(x_a).data.cpu().numpy()
		#	#x_c = Variable(torch.cat((a.data,torch.from_numpy(s)),1), requires_grad=False).unsqueeze(0)
		#	#q = critic.forward(x_c)
		#elif net_type == 4:
		#	inp = np.concatenate((s.state,np.asarray([phase])))
		#	x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
		#	a = actor.forward(x_a).data.cpu().numpy()
		#	#x_c = Variable(torch.cat((a.data,torch.from_numpy(inp)),1).type(dtype), requires_grad=False).unsqueeze(0)
		#	#q = critic.forward(x_c)

		s_prime, reward, terminal, info = env.step(a)
		total_reward += reward
		step_count += 1
		if step_count >= 1000:
			print 'Episode length limit exceeded in greedy!'
			break
		s = s_prime

	print 'Total reward', total_reward
	print 'Number of steps', step_count

	f.close()

if __name__ == '__main__':
	main()
