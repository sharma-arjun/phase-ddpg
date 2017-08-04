import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

def kn(p, n):
	return int((math.floor((4*p)/(2*math.pi)) + n - 1) % 4)

def spline_w(p):
	return ((4*p)/(2*math.pi)) % 1
	

class Alpha(object):
	def __init__(self, n_layers=1):
		self._parameters = {}
		self._parameters['weight_0'] = None
		self._parameters['bias_0'] = None
		if n_layers == 2:
			self._parameters['weight_1'] = None
			self._parameters['bias_1'] = None


		self._parameters['weight'] = None
		self._parameters['bias'] = None

		self._grad = {}
		self._grad['weight_0'] = None
		self._grad['bias_0'] = None
		if n_layers == 2:
			self._grad['weight_1'] = None
			self._grad['bias_1'] = None

		self._grad['weight'] = None
		self._grad['bias'] = None

def init_fanin(tensor):
	fanin = tensor.size(1)
	v = 1.0 / np.sqrt(fanin)
	init.uniform(tensor, -v, v)

class PMLP(nn.Module):
	def __init__(self, input_size, output_size, hidden_size, dtype, n_layers=1, batch_size=1, scale=1.0, tanh_flag=0):
		super(PMLP, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.batch_size = batch_size
		self.scale = scale # scale output of actor from [-1,1] to range of action space [-scale,scale]. set to 1 for critic
		self.tanh_flag = tanh_flag # 1 for actor, 0 for critic (since critic range need not be restricted to [-1,1])
                self.dtype = dtype

		#self.control_gru_list = []
		self.control_hidden_list = []
		self.control_h2o_list = []

	
		#self.gru_00 = nn.GRUCell(self.input_size, self.hidden_size)
		self.l_00 = nn.Linear(self.input_size, self.hidden_size).type(dtype)
		self.h2o_0 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		#self.gru_10 = nn.GRUCell(self.input_size, self.hidden_size)
		self.l_10 = nn.Linear(self.input_size, self.hidden_size).type(dtype)
		self.h2o_1 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		#self.gru_20 = nn.GRUCell(self.input_size, self.hidden_size)
		self.l_20 = nn.Linear(self.input_size, self.hidden_size).type(dtype)
		self.h2o_2 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		#self.gru_30 = nn.GRUCell(self.input_size, self.hidden_size)
		self.l_30 = nn.Linear(self.input_size, self.hidden_size).type(dtype)
		self.h2o_3 = nn.Linear(self.hidden_size, self.output_size).type(dtype)

                init_fanin(self.l_00.weight)
                init_fanin(self.l_10.weight)
                init_fanin(self.l_20.weight)
                init_fanin(self.l_30.weight)

                init.uniform(self.h2o_0.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_0.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_1.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_1.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_2.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_2.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_3.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_3.bias,-3e-3, 3e-3)

		if n_layers == 2:
			
			#self.gru_01 = nn.GRUCell(self.hidden_size, self.hidden_size)
			#self.gru_11 = nn.GRUCell(self.hidden_size, self.hidden_size)
			#self.gru_21 = nn.GRUCell(self.hidden_size, self.hidden_size)
			#self.gru_31 = nn.GRUCell(self.hidden_size, self.hidden_size)
			self.l_01 = nn.Linear(self.hidden_size, self.hidden_size).type(dtype)
			self.l_11 = nn.Linear(self.hidden_size, self.hidden_size).type(dtype)
			self.l_21 = nn.Linear(self.hidden_size, self.hidden_size).type(dtype)
			self.l_31 = nn.Linear(self.hidden_size, self.hidden_size).type(dtype)

                        init_fanin(self.l_01.weight)
                        init_fanin(self.l_11.weight)
                        init_fanin(self.l_21.weight)
                        init_fanin(self.l_31.weight)

		self.control_hidden_list.append([self.l_00, self.l_10, self.l_20, self.l_30])
		if n_layers == 2:
			self.control_hidden_list.append([self.l_01, self.l_11, self.l_21, self.l_31])

		self.control_h2o_list = [self.h2o_0, self.h2o_1, self.h2o_2, self.h2o_3]

		self.alpha = []
		for i in range(4):
			self.alpha.append(Alpha(n_layers))

		self.init_controls(self.control_hidden_list, self.control_h2o_list, self.alpha)
		#self.h_0 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)
		#if n_layers == 2:
		#	self.h_1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)

		self.hidden_list = []
		self.h2o_list = []
		self.phase_list = []

		# to initialize grad of control hidden and h2o ... I need to do this stupid thing ...
		dummy_x = Variable(torch.zeros(batch_size, input_size), requires_grad=False).type(dtype)
		dummy_y = Variable(torch.zeros(batch_size, output_size), requires_grad=False).type(dtype)
		dummy_criterion = nn.MSELoss()

		if n_layers == 1:
			for l, h2o in zip(self.control_hidden_list[0], self.control_h2o_list):
				dummy_h = F.relu(l(dummy_x))
				dummy_o = h2o(dummy_h)
				dummy_loss = dummy_criterion(dummy_o, dummy_y)
				dummy_loss.backward()

		if n_layers == 2:
			for l0, l1, h2o in zip(self.control_hidden_list[0], self.control_hidden_list[1], self.control_h2o_list):
				dummy_h0 = F.relu(l0(dummy_x))
				dummy_h1 = l1(dummy_h0)
				dummy_o = h2o(dummy_h1)
				dummy_loss = dummy_criterion(dummy_o, dummy_y)
				dummy_loss.backward()

		# reset to zero after dummy pass
		#self.h_0 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)
		#if n_layers == 2:
		#	self.h_1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)

	def forward(self,x,phase):
		w = self.weight_from_phase(phase, self.alpha)
		hiddens = []
		l0 = nn.Linear(self.input_size, self.hidden_size).type(self.dtype)
		hiddens.append(l0)
		h2o = nn.Linear(self.hidden_size, self.output_size).type(self.dtype)
		if self.n_layers == 2:
			l1 = nn.Linear(self.hidden_size, self.hidden_size).type(self.dtype)
			hiddens.append(l1)

		self.set_weight(w, hiddens, h2o)
		self.hidden_list.append(hiddens)
		self.h2o_list.append(h2o)
		self.phase_list.append(phase)

		h_0 = F.relu(l0(x))
		if self.n_layers == 2:
			h_1 = F.relu(l1(h_0))
			if self.tanh_flag:
				o = F.tanh(h2o(h_1))
			else:
				o = h2o(h_1)
		else:
			if self.tanh_flag:
				o = F.tanh(h2o(h_0))
			else:
				o = h2o(h_0)

		return self.scale*o

	def reset(self):
		#self.h_0 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True)
		#if self.n_layers == 2:
		#	self.h_1 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True)
		
		self.hidden_list = []
		self.h2o_list = []
		self.phase_list = []

		self.init_controls(self.control_hidden_list, self.control_h2o_list, self.alpha)


	def weight_from_phase(self, phase, alpha):
		weight = {}
		w = spline_w(phase)
		for key in alpha[0]._parameters.keys():
			weight[key] = alpha[kn(phase, 1)]._parameters[key] + w*0.5*(alpha[kn(phase, 2)]._parameters[key] - alpha[kn(phase, 0)]._parameters[key]) + w*w*(alpha[kn(phase, 0)]._parameters[key] - 2.5*alpha[kn(phase, 1)]._parameters[key] + 2*alpha[kn(phase, 2)]._parameters[key] - 0.5*alpha[kn(phase, 3)]._parameters[key]) + w*w*w*(1.5*alpha[kn(phase, 1)]._parameters[key] - 1.5*alpha[kn(phase, 2)]._parameters[key] + 0.5*alpha[kn(phase, 3)]._parameters[key] - 0.5*alpha[kn(phase, 0)]._parameters[key])

		return weight


	def set_weight(self, w, hiddens, h2o):
		count = 0
		for l in hiddens:
			l._parameters['weight'].data = w['weight_' + str(count)]
			l._parameters['bias'].data = w['bias_' + str(count)]
			count += 1

		h2o._parameters['weight'].data = w['weight']
		h2o._parameters['bias'].data = w['bias']

	def init_controls(self, list_of_hidden, list_of_h2o, alpha):
		for i in range(len(alpha)):
			for j in range(len(list_of_hidden)):
				l = list_of_hidden[j][i]
				alpha[i]._parameters['weight_' + str(j)] = l._parameters['weight'].data.clone()
				alpha[i]._parameters['bias_' + str(j)] = l._parameters['bias'].data.clone()

				#initialize alpha grads as zero here using shape ...
				alpha[i]._grad['weight_' + str(j)] = torch.zeros(l._parameters['weight'].data.size()).type(self.dtype)
				alpha[i]._grad['bias_' + str(j)] = torch.zeros(l._parameters['bias'].data.size()).type(self.dtype)

			h2o = list_of_h2o[i]
			alpha[i]._parameters['weight'] = h2o._parameters['weight'].data.clone()
			alpha[i]._parameters['bias'] = h2o._parameters['bias'].data.clone()
			alpha[i]._grad['weight'] = torch.zeros(h2o._parameters['weight'].data.size()).type(self.dtype)
			alpha[i]._grad['bias'] = torch.zeros(h2o._parameters['bias'].data.size()).type(self.dtype)


	def update_control_gradients(self):
		for hiddens, phase in zip(self.hidden_list, self.phase_list):
			w = spline_w(phase)
			count = 0
			for l in hiddens:
				for key in l._parameters.keys():
					self.alpha[kn(phase,0)]._grad[key + '_' + str(count)] += l._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
					self.alpha[kn(phase,1)]._grad[key + '_' + str(count)] += l._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
					self.alpha[kn(phase,2)]._grad[key + '_' + str(count)] += l._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
					self.alpha[kn(phase,3)]._grad[key + '_' + str(count)] += l._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)
				count += 1

		for h2o, phase in zip(self.h2o_list, self.phase_list):
			w = spline_w(phase)
			for key in h2o._parameters.keys():
				self.alpha[kn(phase,0)]._grad[key] += h2o._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
				self.alpha[kn(phase,1)]._grad[key] += h2o._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
				self.alpha[kn(phase,2)]._grad[key] += h2o._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
				self.alpha[kn(phase,3)]._grad[key] += h2o._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)


		for i in range(len(self.control_hidden_list)):
			for alpha, l in zip(self.alpha, self.control_hidden_list[i]):
				for key in l._parameters.keys():
					l._parameters[key].grad.data += alpha._grad[key + '_' + str(i)]

		for alpha, h2o in zip(self.alpha, self.control_h2o_list):
			for key in h2o._parameters.keys():
				h2o._parameters[key].grad.data += alpha._grad[key]
