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
	

#class Alpha(object):
#	def __init__(self, n_layers=1):
#		self._parameters = {}
#		self._parameters['weight_ih_0'] = None
#		self._parameters['weight_hh_0'] = None
#		self._parameters['bias_ih_0'] = None
#		self._parameters['bias_hh_0'] = None
#		if n_layers == 2:
#			self._parameters['weight_ih_1'] = None
#			self._parameters['weight_hh_1'] = None
#			self._parameters['bias_ih_1'] = None
#			self._parameters['bias_hh_1'] = None


#		self._parameters['weight'] = None
#		self._parameters['bias'] = None

#		self._grad = {}
#		self._grad['weight_ih_0'] = None
#		self._grad['weight_hh_0'] = None
#		self._grad['bias_ih_0'] = None
#		self._grad['bias_hh_0'] = None
#		if n_layers == 2:
#			self._grad['weight_ih_1'] = None
#			self._grad['weight_hh_1'] = None
#			self._grad['bias_ih_1'] = None
#			self._grad['bias_hh_1'] = None

#		self._grad['weight'] = None
#		self._grad['bias'] = None

class PLSTM(nn.Module):
	def __init__(self, input_size, output_size, hidden_size, dtype, n_layers=1, batch_size=1, scale=1.0, tanh_flag=0):
		super(PLSTM, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.batch_size = batch_size
		self.scale = scale
		self.tanh_flag = tanh_flag
		self.dtype = dtype

		self.control_gru_list = []
		self.control_h2o_list = []

	
		self.gru_00 = nn.GRUCell(self.input_size, self.hidden_size).type(dtype)
		self.h2o_0 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		self.gru_10 = nn.GRUCell(self.input_size, self.hidden_size).type(dtype)
		self.h2o_1 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		self.gru_20 = nn.GRUCell(self.input_size, self.hidden_size).type(dtype)
		self.h2o_2 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		self.gru_30 = nn.GRUCell(self.input_size, self.hidden_size).type(dtype)
		self.h2o_3 = nn.Linear(self.hidden_size, self.output_size).type(dtype)

                init.uniform(self.h2o_0.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_0.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_1.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_1.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_2.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_2.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_3.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_3.bias,-3e-3, 3e-3)

		if n_layers == 2:
			
			self.gru_01 = nn.GRUCell(self.hidden_size, self.hidden_size).type(dtype)
			self.gru_11 = nn.GRUCell(self.hidden_size, self.hidden_size).type(dtype)
			self.gru_21 = nn.GRUCell(self.hidden_size, self.hidden_size).type(dtype)
			self.gru_31 = nn.GRUCell(self.hidden_size, self.hidden_size).type(dtype)
		
		self.control_gru_list.append([self.gru_00, self.gru_10, self.gru_20, self.gru_30])
		if n_layers == 2:
			self.control_gru_list.append([self.gru_01, self.gru_11, self.gru_21, self.gru_31])

		self.control_h2o_list = [self.h2o_0, self.h2o_1, self.h2o_2, self.h2o_3]

		#self.alpha = []
		#for i in range(4):
		#	self.alpha.append(Alpha(n_layers))

		#self.init_controls(self.control_gru_list, self.control_h2o_list, self.alpha)
		self.h_0 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)
		if n_layers == 2:
			self.h_1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)

		self.gru_list = []
		self.h2o_list = []
		self.phase_list = []

		# to initialize grad of control gru and h2o ... I need to do this stupid thing ...
		dummy_x = Variable(torch.zeros(batch_size, input_size), requires_grad=False).type(dtype)
		dummy_y = Variable(torch.zeros(batch_size, output_size), requires_grad=False).type(dtype)
		dummy_criterion = nn.MSELoss()

		if n_layers == 1:
			for gru, h2o in zip(self.control_gru_list[0], self.control_h2o_list):
				dummy_h = gru(dummy_x, self.h_0)
				dummy_o = h2o(dummy_h)
				dummy_loss = dummy_criterion(dummy_o, dummy_y)
				dummy_loss.backward()

		if n_layers == 2:
			for gru0, gru1, h2o in zip(self.control_gru_list[0], self.control_gru_list[1], self.control_h2o_list):
				dummy_h0 = gru0(dummy_x, self.h_0)
				dummy_h1 = gru1(dummy_h0, self.h_1)
				dummy_o = h2o(dummy_h1)
				dummy_loss = dummy_criterion(dummy_o, dummy_y)
				dummy_loss.backward()

		# reset to zero after dummy pass
		self.h_0 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)
		if n_layers == 2:
			self.h_1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)

	def forward(self,x,phase):
		w = self.weight_from_phase(phase, self.control_gru_list, self.control_h2o_list)
		grus = []
		gru0 = nn.GRUCell(self.input_size, self.hidden_size)
		grus.append(gru0)
		h2o = nn.Linear(self.hidden_size, self.output_size)
		if self.n_layers == 2:
			gru1 = nn.GRUCell(self.hidden_size, self.hidden_size)
			grus.append(gru1)

		self.set_weight(w, grus, h2o)	
		self.gru_list.append(grus)
		self.h2o_list.append(h2o)
		self.phase_list.append(phase)

		self.h_0 = gru0(x, self.h_0)
		if self.n_layers == 2:
			self.h_1 = gru1(self.h_0, self.h_1)
			if self.tanh_flag:
				o = F.tanh(h2o(self.h_1))
			else:
				o = h2o(self.h_1)
		else:
			if self.tanh_flag:
				o = F.tanh(h2o(self.h_0))
			else:
				o = h2o(self.h_0)

		return self.scale*o

	def reset(self):
		self.h_0 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
		if self.n_layers == 2:
			self.h_1 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
		
		self.gru_list = []
		self.h2o_list = []
		self.phase_list = []

		#self.init_controls(self.control_gru_list, self.control_h2o_list, self.alpha)


	def weight_from_phase(self, phase, control_gru_list, control_h2o_list):
		weight = {}
		w = spline_w(phase)
                for n in range(len(control_gru_list)):
        		for key in control_gru_list[0][0]._parameters.keys():
	        		weight[key+'_' + str(n)] = control_gru_list[n][kn(phase, 1)]._parameters[key].data + w*0.5*(control_gru_list[n][kn(phase, 2)]._parameters[key].data - control_gru_list[n][kn(phase, 0)]._parameters[key].data) + w*w*(control_gru_list[n][kn(phase, 0)]._parameters[key].data - 2.5*control_gru_list[n][kn(phase, 1)]._parameters[key].data + 2*control_gru_list[n][kn(phase, 2)]._parameters[key].data - 0.5*control_gru_list[n][kn(phase, 3)]._parameters[key].data) + w*w*w*(1.5*control_gru_list[n][kn(phase, 1)]._parameters[key].data - 1.5*control_gru_list[n][kn(phase, 2)]._parameters[key].data + 0.5*control_gru_list[n][kn(phase, 3)]._parameters[key].data - 0.5*control_gru_list[n][kn(phase, 0)]._parameters[key].data)

        	for key in control_h2o_list[0]._parameters.keys():
	                weight[key] = control_h2o_list[kn(phase, 1)]._parameters[key].data + w*0.5*(control_h2o_list[kn(phase, 2)]._parameters[key].data - control_h2o_list[kn(phase, 0)]._parameters[key].data) + w*w*(control_h2o_list[kn(phase, 0)]._parameters[key].data - 2.5*control_h2o_list[kn(phase, 1)]._parameters[key].data + 2*control_h2o_list[kn(phase, 2)]._parameters[key].data - 0.5*control_h2o_list[kn(phase, 3)]._parameters[key].data) + w*w*w*(1.5*control_h2o_list[kn(phase, 1)]._parameters[key].data - 1.5*control_h2o_list[kn(phase, 2)]._parameters[key].data + 0.5*control_h2o_list[kn(phase, 3)]._parameters[key].data - 0.5*control_h2o_list[kn(phase, 0)]._parameters[key].data)

		return weight


	def set_weight(self, w, grus, h2o):
		count = 0
		for gru in grus:
			gru._parameters['weight_ih'].data = w['weight_ih_' + str(count)]
			gru._parameters['weight_hh'].data = w['weight_hh_' + str(count)]
			gru._parameters['bias_ih'].data = w['bias_ih_' + str(count)]
			gru._parameters['bias_hh'].data = w['bias_hh_' + str(count)]
			count += 1
			gru.type(self.dtype)

		h2o._parameters['weight'].data = w['weight']
		h2o._parameters['bias'].data = w['bias']
		h2o.type(self.dtype)

	#def init_controls(self, list_of_gru, list_of_h2o, alpha):
	#	for i in range(len(alpha)):
	#		for j in range(len(list_of_gru)):
	#			gru = list_of_gru[j][i]
	#			alpha[i]._parameters['weight_ih_' + str(j)] = gru._parameters['weight_ih'].data.clone()
	#			alpha[i]._parameters['weight_hh_' + str(j)] = gru._parameters['weight_hh'].data.clone()
	#			alpha[i]._parameters['bias_ih_' + str(j)] = gru._parameters['bias_ih'].data.clone()
	#			alpha[i]._parameters['bias_hh_' + str(j)] = gru._parameters['bias_hh'].data.clone()

	#			#initialize alpha grads as zero here using shape ...
	#			alpha[i]._grad['weight_ih_' + str(j)] = torch.zeros(gru._parameters['weight_ih'].data.size()).type(self.dtype)
	#			alpha[i]._grad['weight_hh_' + str(j)] = torch.zeros(gru._parameters['weight_hh'].data.size()).type(self.dtype)
	#			alpha[i]._grad['bias_ih_' + str(j)] = torch.zeros(gru._parameters['bias_ih'].data.size()).type(self.dtype)
	#			alpha[i]._grad['bias_hh_' + str(j)] = torch.zeros(gru._parameters['bias_hh'].data.size()).type(self.dtype)

	#		h2o = list_of_h2o[i]
	#		alpha[i]._parameters['weight'] = h2o._parameters['weight'].data.clone()
	#		alpha[i]._parameters['bias'] = h2o._parameters['bias'].data.clone()
	#		alpha[i]._grad['weight'] = torch.zeros(h2o._parameters['weight'].data.size()).type(self.dtype)
	#		alpha[i]._grad['bias'] = torch.zeros(h2o._parameters['bias'].data.size()).type(self.dtype)


	def update_control_gradients(self):
		for grus, phase in zip(self.gru_list, self.phase_list):
			w = spline_w(phase)
			count = 0
			for gru in grus:
				for key in gru._parameters.keys():
					self.control_gru_list[count][kn(phase,0)]._parameters[key].grad.data += gru._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
					self.control_gru_list[count][kn(phase,1)]._parameters[key].grad.data += gru._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
					self.control_gru_list[count][kn(phase,2)]._parameters[key].grad.data += gru._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
					self.control_gru_list[count][kn(phase,3)]._parameters[key].grad.data += gru._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)
				count += 1

		for h2o, phase in zip(self.h2o_list, self.phase_list):
			w = spline_w(phase)
			for key in h2o._parameters.keys():
				self.control_h2o_list[kn(phase,0)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
				self.control_h2o_list[kn(phase,1)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
				self.control_h2o_list[kn(phase,2)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
				self.control_h2o_list[kn(phase,3)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)


		#for i in range(len(self.control_gru_list)):
		#	for alpha, gru in zip(self.alpha, self.control_gru_list[i]):
		#		for key in gru._parameters.keys():
		#			gru._parameters[key].grad.data += alpha._grad[key + '_' + str(i)]

		#for alpha, h2o in zip(self.alpha, self.control_h2o_list):
		#	for key in h2o._parameters.keys():
		#		h2o._parameters[key].grad.data += alpha._grad[key]
