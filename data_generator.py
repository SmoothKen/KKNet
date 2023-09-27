"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - data_generator file

This file contains the dataset and data generator classes

"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from util import index2centf
from feature_extraction import get_CenFreq

def reorganize(x, octave_res):
	n_order = []
	max_bin = x.shape[1]
	for i in range(octave_res):
		n_order += [j for j in range(i, max_bin, octave_res)]
	nx = [x[:,n_order[i],:] for i in range(x.shape[1])]
	nx = np.array(nx)
	nx = nx.transpose((1,0,2))
	return nx
	 

class TONetTrainDataset(Dataset): 
	def __init__(self, data_list, config):
		self.config = config 
		# self.cfp_dir = os.path.join(config.data_path,config.cfp_dir)
		# self.f0_dir = os.path.join(config.data_path,"f0ref")
		self.cfp_dir = "/home/ken/Downloads/cfp_saved/"
		self.f0_dir = "/home/ken/Downloads/labels_and_waveform/"
		self.data_list = data_list
		self.cent_f = np.array(get_CenFreq(config.startfreq, config.stopfreq, config.octave_res))
		# init data array
		self.data_cfp = []
		self.data_gd = []
		self.data_tcfp = []
		seg_frame = config.seg_frame
		shift_frame = config.shift_frame
		print("Data List:", data_list)
		with open(data_list, "r") as f:
			data_txt = f.readlines()
			data_txt = [d.split(".")[0] for d in data_txt]
		# data_txt = data_txt[:100]
		print("Song Size:", len(data_txt))
		# process cfp
		for i, filename in enumerate(tqdm(data_txt)):
			# file set
			cfp_file = os.path.join(self.cfp_dir, filename + ".npy")
			ref_file = os.path.join(self.f0_dir, filename + ".txt")
			# get raw cfp and freq
			temp_cfp = np.load(cfp_file, allow_pickle = True)
			# temp_cfp[0, :, :] = temp_cfp[1, :, :] * temp_cfp[2, :, :]
			temp_freq = np.loadtxt(ref_file)
			temp_freq = temp_freq[:,1]
			# check length
			if temp_freq.shape[0] > temp_cfp.shape[2]:
				temp_freq = temp_freq[:temp_cfp.shape[2]]
			else:
				temp_cfp = temp_cfp[:,:,:temp_freq.shape[0]]
			# build data
			for j in range(0, temp_cfp.shape[2], shift_frame): 
				bgnt = j
				endt = j + seg_frame
				# temp_x = temp_cfp[:, :, bgnt:endt]
				temp_gd = index2centf(temp_freq[bgnt:endt], self.cent_f)
				
				# left and right pad temp_x to counter shrinking
				# we hope that bgnt - network_time_shrink_size >= 0 and endt + network_time_shrink_size <= temp_cfp.shape[2]
				from config import network_time_shrink_size
				temp_x = temp_cfp[:, :, max(0, bgnt - network_time_shrink_size):min(endt + network_time_shrink_size, temp_cfp.shape[2])]
				
				# print(temp_x.shape[2])
				
				if bgnt - network_time_shrink_size < 0:
					left_padding_size = abs(bgnt - network_time_shrink_size)
					temp_x = np.concatenate([np.zeros((temp_cfp.shape[0], temp_cfp.shape[1], left_padding_size)), temp_x], axis = 2)
					
				# print(temp_x.shape[2])
				if endt + network_time_shrink_size > temp_cfp.shape[2]:
					# in this temp_gds will have everything at the right end
					if endt >= temp_cfp.shape[2]:
						right_padding_size = network_time_shrink_size
					else:
						right_padding_size = endt + network_time_shrink_size - temp_cfp.shape[2]
						
					temp_x = np.concatenate([temp_x, np.zeros((temp_cfp.shape[0], temp_cfp.shape[1], right_padding_size))], axis = 2)
					# print(right_padding_size, endt, temp_cfp.shape[2])
					
				# print(temp_x.shape[2], len(temp_gd), 2*network_time_shrink_size)
				
				
				if temp_x.shape[2] < seg_frame + 2*network_time_shrink_size:
					rl = temp_x.shape[2]
					# pad_x = np.zeros((temp_x.shape[0], temp_x.shape[1], seg_frame))
					pad_x = np.zeros((temp_x.shape[0], temp_x.shape[1], seg_frame + 2*network_time_shrink_size))
					pad_gd = np.zeros(seg_frame)
					# pad_gd[:rl] = temp_gd
					pad_gd[:rl - 2*network_time_shrink_size] = temp_gd
					pad_x[:,:, :rl] = temp_x
					temp_x = pad_x
					temp_gd = pad_gd
				
				assert temp_x.shape[2] - len(temp_gd) == 2*network_time_shrink_size	
				
					
				temp_tx = reorganize(temp_x[:], config.octave_res)
				# self.data_tcfp.append(temp_tx)
				# to save memory
				self.data_tcfp = list(range(50000))
				self.data_cfp.append(temp_x)
				
				
				# print(temp_gd.shape, temp_freq[bgnt:endt].shape)
				self.data_gd.append(temp_gd)
		self.data_cfp = np.array(self.data_cfp)
		self.data_tcfp = np.array(self.data_tcfp)
		# no need for tcfp for now (to save space)
		self.data_gd = np.array(self.data_gd)
		print("Total Datasize:", self.data_cfp.shape)
					   
	def __len__(self):
		return len(self.data_cfp)
	
	def __getitem__(self,index):
		temp_dict = {
			"cfp": self.data_cfp[index].astype(np.float32),
			"tcfp": self.data_tcfp[index].astype(np.float32),
			"gd": self.data_gd[index]
		}
		# print("Haaa", temp_dict["gd"].shape)
		return temp_dict


class TONetTestDataset(Dataset): 
	def __init__(self, data_list, config):
		self.config = config 
		# self.cfp_dir = os.path.join(config.data_path,config.cfp_dir)
		# self.f0_dir = os.path.join(config.data_path,"f0ref")
		self.cfp_dir = "/home/ken/Downloads/cfp_saved/"
		self.f0_dir = "/home/ken/Downloads/labels_and_waveform/"
		
		self.data_list = data_list
		self.cent_f = np.array(get_CenFreq(config.startfreq, config.stopfreq, config.octave_res))
		# init data array
		self.data_names = []
		self.data_cfp = []
		self.data_gd = []
		self.data_len = []
		self.data_tcfp = []
		seg_frame = config.seg_frame
		shift_frame = config.shift_frame
		print("Data List:", data_list)
		with open(data_list, "r") as f:
			data_txt = f.readlines()
			data_txt = [d.split(".")[0] for d in data_txt]
		print("Song Size:", len(data_txt))
		# process cfp
		for i, filename in enumerate(tqdm(data_txt)):
			
			group_cfp = []
			group_gd = []
			group_tcfp = []
			# file set
			cfp_file = os.path.join(self.cfp_dir, filename + ".npy")
			ref_file = os.path.join(self.f0_dir, filename + ".txt")
			
			
			# get raw cfp and freq
			temp_cfp = np.load(cfp_file, allow_pickle = True)
			# temp_cfp[0, :, :] = temp_cfp[1, :, :] * temp_cfp[2, :, :]
			temp_freq = np.loadtxt(ref_file)
			temp_freq = temp_freq[:,1]
			self.data_len.append(len(temp_freq))
			# check length
			if temp_freq.shape[0] > temp_cfp.shape[2]:
				temp_freq = temp_freq[:temp_cfp.shape[2]]
			else:
				temp_cfp = temp_cfp[:,:,:temp_freq.shape[0]]
			# build data
			for j in range(0, temp_cfp.shape[2], shift_frame): 
				bgnt = j
				endt = j + seg_frame
				# temp_x = temp_cfp[:, :, bgnt:endt]
				temp_gd = temp_freq[bgnt:endt]
				
				
				# left and right pad temp_x to counter shrinking
				# we hope that bgnt - network_time_shrink_size >= 0 and endt + network_time_shrink_size <= temp_cfp.shape[2]
				from config import network_time_shrink_size
				temp_x = temp_cfp[:, :, max(0, bgnt - network_time_shrink_size):min(endt + network_time_shrink_size, temp_cfp.shape[2])]
				
				if bgnt - network_time_shrink_size < 0:
					left_padding_size = abs(bgnt - network_time_shrink_size)
					temp_x = np.concatenate([np.zeros((temp_cfp.shape[0], temp_cfp.shape[1], left_padding_size)), temp_x], axis = 2)
					
				if endt + network_time_shrink_size > temp_cfp.shape[2]:
					# in this temp_gds will have everything at the right end
					if endt >= temp_cfp.shape[2]:
						right_padding_size = network_time_shrink_size
					else:
						right_padding_size = endt + network_time_shrink_size - temp_cfp.shape[2]
						
					temp_x = np.concatenate([temp_x, np.zeros((temp_cfp.shape[0], temp_cfp.shape[1], right_padding_size))], axis = 2)
					
				# print(temp_x.shape[2], len(temp_gd), 2*network_time_shrink_size)
				
				# not enough only when we are already at the right end, hence padding gds by 0, it will correspond to white padding which is also 0
				if temp_x.shape[2] < seg_frame + 2*network_time_shrink_size:
					rl = temp_x.shape[2]
					# pad_x = np.zeros((temp_x.shape[0], temp_x.shape[1], seg_frame))
					pad_x = np.zeros((temp_x.shape[0], temp_x.shape[1], seg_frame + 2*network_time_shrink_size))
					pad_gd = np.zeros(seg_frame)
					# pad_gd[:rl] = temp_gd
					pad_gd[:rl - 2*network_time_shrink_size] = temp_gd
					pad_x[:,:, :rl] = temp_x
					temp_x = pad_x
					temp_gd = pad_gd
				
				assert temp_x.shape[2] - len(temp_gd) == 2*network_time_shrink_size	
				
				
				temp_tx = reorganize(temp_x[:], config.octave_res)
				group_tcfp.append(temp_tx)
				group_cfp.append(temp_x)
				group_gd.append(temp_gd)
			group_tcfp = np.array(group_tcfp)
			group_cfp = np.array(group_cfp)
			group_gd = np.array(group_gd)
			
			self.data_names.append(ref_file)
			self.data_tcfp.append(group_tcfp)
			self.data_cfp.append(group_cfp)
			self.data_gd.append(group_gd)
					   
	def __len__(self):
		return len(self.data_cfp)
	
	def __getitem__(self,index):
		temp_dict = {
			"cfp": self.data_cfp[index].astype(np.float32),
			"tcfp": self.data_tcfp[index].astype(np.float32),
			"gd": self.data_gd[index],
			"length": self.data_len[index],
			"name": self.data_names[index]
		}
		return temp_dict
