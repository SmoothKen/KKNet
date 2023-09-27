import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
	"""Initialize a Linear or Convolutional layer. """
	nn.init.xavier_uniform_(layer.weight)
 
	if hasattr(layer, 'bias'):
		if layer.bias is not None:
			layer.bias.data.fill_(0.)
			
	
def init_bn(bn):
	"""Initialize a Batchnorm layer. """
	bn.bias.data.fill_(0.)
	bn.weight.data.fill_(1.)


def init_gru(rnn):
	"""Initialize a GRU layer. """
	
	def _concat_init(tensor, init_funcs):
		(length, fan_out) = tensor.shape
		fan_in = length // len(init_funcs)
	
		for (i, init_func) in enumerate(init_funcs):
			init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
		
	def _inner_uniform(tensor):
		fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
		nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
	
	for i in range(rnn.num_layers):
		_concat_init(
			getattr(rnn, 'weight_ih_l{}'.format(i)),
			[_inner_uniform, _inner_uniform, _inner_uniform]
		)
		torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

		_concat_init(
			getattr(rnn, 'weight_hh_l{}'.format(i)),
			[_inner_uniform, _inner_uniform, nn.init.orthogonal_]
		)
		torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, momentum):
		
		super(ConvBlock, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=in_channels, 
							  out_channels=out_channels,
							  kernel_size=(3, 5), stride=(1, 1),
							  padding=(0, 1), bias=False)
							  
		self.conv2 = nn.Conv2d(in_channels=out_channels, 
							  out_channels=out_channels,
							  kernel_size=(3, 3), stride=(1, 1),
							  padding=(0, 1), bias=False)

		self.bn1 = nn.BatchNorm2d(out_channels, momentum)
		self.bn2 = nn.BatchNorm2d(out_channels, momentum)

		self.init_weight()
		
	def init_weight(self):
		init_layer(self.conv1)
		init_layer(self.conv2)
		init_bn(self.bn1)
		init_bn(self.bn2)

		
	def forward(self, input, pool_size=(2, 2), pool_type='avg'):
		"""
		Args:
		  input: (batch_size, in_channels, time_steps, freq_bins)

		Outputs:
		  output: (batch_size, out_channels, classes_num)
		"""

		x = F.relu_(self.bn1(self.conv1(input)))		
		x = F.relu_(self.bn2(self.conv2(x)))
		
		# x = F.selu(self.conv1(input))
		# x = F.selu(self.conv2(x))
		
		if pool_type == 'avg':
			x = F.avg_pool2d(x, kernel_size=pool_size)
		
		return x


if torch.cuda.is_available():
	device = torch.device("cuda")  
	print("Using cuda")
else:
	device = torch.device("cpu")
	print("Using cpu")


from config import include_model_tweak
class AcousticModelCRnn8Dropout(nn.Module):
	def __init__(self, classes_num = 361, midfeat = 2560, momentum = 0.01):
		super(AcousticModelCRnn8Dropout, self).__init__()

		self.conv_block1 = ConvBlock(in_channels=3, out_channels=48, momentum=momentum)
		self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
		self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
		self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

		self.fc5 = nn.Linear(midfeat, 768, bias=False)
		self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

		self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
			bias=True, batch_first=True, dropout=0., bidirectional=True)

		self.fc = nn.Linear(512, classes_num, bias=True)
		
		self.sfmax = torch.nn.Softmax(dim = 2)
		self.init_weight()

	def init_weight(self):
		init_layer(self.fc5)
		init_bn(self.bn5)
		init_gru(self.gru)
		init_layer(self.fc)

	def forward(self, x):
		"""
		Args:
		  input: (batch_size, channels_num, time_steps, freq_bins)

		Outputs:
		  output: (batch_size, time_steps, classes_num)
		"""

		x = self.conv_block1(x.transpose(2,3), pool_size=(1, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)

		x = x.transpose(1, 2).flatten(2)

		x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
		x = F.dropout(x, p=0.5, training=self.training, inplace=False)

		(x, _) = self.gru(x)
		x = F.dropout(x, p=0.5, training=self.training, inplace=False)
		x = self.fc(x)

		
		if include_model_tweak:
			# x[:, :, -1] = np.log(x.shape[-1]) - x[:, :, -1]
			# x[:, :, 0] = np.log(49*(x.shape[-1]-1)) - x[:, :, 0]
			x[:, :, 0] = 1 - x[:, :, 0]


		output = self.sfmax(x)
		return output.transpose(1,2), None




if __name__ == "__main__":
	# (batch_size, 3 -> CFP, 360 -> FREQ_BINS, 144 -> TIME_STEPS)
	x = torch.randn(2, 3, 360, 144, device = device)
	print(AcousticModelCRnn8Dropout().to(device)(x)[0].shape)

