import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init


def init_fanin(tensor):
	fanin = tensor.size(1)
	v = 1.0 / np.sqrt(fanin)
	init.uniform(tensor, -v, v)

class MLP(nn.Module):
	def __init__(self, input_size_state, input_size_action, output_size, hidden_size, n_layers=1, scale=1.0, tanh_flag=0):
		super(MLP, self).__init__()

		self.input_size_state = input_size_state
                self.input_size_action = input_size_action
		self.output_size = output_size
		self.hidden_size_1 = hidden_size[0]
                self.hidden_size_2 = hidden_size[-1]
		self.n_layers = n_layers
		self.scale = scale
		self.tanh_flag = tanh_flag

		self.l1 = nn.Linear(self.input_size_state, self.hidden_size_1)
		init_fanin(self.l1.weight)
		if n_layers == 2:
			self.l2 = nn.Linear(self.hidden_size_1 + self.input_size_action, self.hidden_size_2)
			init_fanin(self.l2.weight)

		self.h2o = nn.Linear(self.hidden_size_2, self.output_size)
		init.uniform(self.h2o.weight,-3e-3, 3e-3)
		init.uniform(self.h2o.bias,-3e-3, 3e-3)


	def forward(self,x):
                x_a = x[:,-self.input_size_action:]
                x_s = x[:,0:-self.input_size_action]
		h0 = F.relu(self.l1(x_s))
		if self.n_layers == 2:
			h1 = F.relu(self.l2(torch.cat((h0,x_a),1)))
			if self.tanh_flag:
				o = F.tanh(self.h2o(h1))
			else:
				o = self.h2o(h1)
		else:
			if self.tanh_flag:
				o = F.tanh(self.h2o(h0))
			else:
				o = self.h2o(h0)

		return self.scale*o

	def reset(self):
		pass
