"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - main file

This file contains the main script

"""
import os
import random
import numpy as np
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import config
from data_generator import TONetTrainDataset, TONetTestDataset
from msnet import MSnet
from tonet import TONet
from multi_dr import MLDRnet
from ftanet import FTAnet
from mcdnn import MCDNN

from util import tonpy_fn


def train():
	train_dataset = TONetTrainDataset(
		data_list = config.train_file,
		config = config
	)
	train_dataloader = DataLoader(
		dataset = train_dataset,
		shuffle = True,
		num_workers = config.n_workers,
		batch_size = config.batch_size,
		drop_last=True
	)
	test_datasets = [
		TONetTestDataset(
			data_list = d,
			config = config
		) for d in config.test_file
	]
	test_dataloaders = [
		DataLoader(
			dataset = d,
			shuffle = False,
			batch_size = 1,
			collate_fn=tonpy_fn
		) for d in test_datasets
	]
	loss_func = nn.BCELoss()

	if config.model_type == "MCDNN":
		me_model = MCDNN()
		me_model_r = MCDNN()
	elif config.model_type == "MLDRNet":
		me_model = MLDRnet()
		me_model_r = MLDRnet()
	elif config.model_type == "FTANet":
		me_model = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
		me_model_r = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
	elif config.model_type == "MSNet":
		me_model = MSnet()
		me_model_r = MSnet()
	else: # AcousticModelCRnn8Dropout
		from piano_net import AcousticModelCRnn8Dropout
		me_model = AcousticModelCRnn8Dropout()
		me_model_r = AcousticModelCRnn8Dropout()
		

	if config.ablation_mode == "single" or config.ablation_mode == "spl" or config.ablation_mode == "spat":
		me_model_r = None 
	model = TONet(
		l_model = me_model,
		r_model = me_model_r,
		config = config,
		loss_func = loss_func,
		mode = config.ablation_mode
	)
	trainer = pl.Trainer(
		# deterministic = True,
		gpus = 1,
		# checkpoint_callback = False,
		max_epochs = config.max_epoch,
		auto_lr_find = True,
		sync_batchnorm=True,
		# check_val_every_n_epoch = 1,
		# val_check_interval = 0.25,
		num_sanity_val_steps=0
	)
	trainer.fit(model, train_dataloader, test_dataloaders)
	

def test():
	test_datasets = [
		TONetTestDataset(
			data_list = d,
			config = config
		) for d in config.test_file
	]
	test_dataloaders = [
		DataLoader(
			dataset = d,
			shuffle = False,
			batch_size = 1,
			collate_fn=tonpy_fn
		) for d in test_datasets
	]
	loss_func = nn.BCELoss()

	if config.model_type == "MCDNN":
		me_model = MCDNN()
		me_model_r = MCDNN()
	elif config.model_type == "MLDRNet":
		me_model = MLDRnet()
		me_model_r = MLDRnet()
	elif config.model_type == "FTANet":
		me_model = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
		me_model_r = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
	elif config.model_type == "MSNet":
		me_model = MSnet()
		me_model_r = MSnet()
	else: # AcousticModelCRnn8Dropout
		from piano_net import AcousticModelCRnn8Dropout
		me_model = AcousticModelCRnn8Dropout()
		me_model_r = AcousticModelCRnn8Dropout()


	if config.ablation_mode == "single" or config.ablation_mode == "spl" or config.ablation_mode == "spat":
		me_model_r = None 
	model = TONet(
		l_model = me_model,
		r_model = me_model_r,
		config = config,
		loss_func = loss_func,
		mode = config.ablation_mode
	)
	trainer = pl.Trainer(
		# deterministic = True,
		gpus = 1,
		# checkpoint_callback = False,
		max_epochs = config.max_epoch,
		auto_lr_find = True,
		sync_batchnorm=True,
		# check_val_every_n_epoch = 1,
		# val_check_interval = 0.25,
	)
	# load the checkpoint
	ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
	model.load_state_dict(ckpt)
	trainer.test(model, test_dataloaders)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog = "TONET for Singing Melody Extraction")
	subparsers = parser.add_subparsers(dest = "mode")
	parser_train = subparsers.add_parser("train")
	parser_test = subparsers.add_parser("test")
	args = parser.parse_args()
	pl.seed_everything(config.random_seed)
	if args.mode == "train":
		train()
	elif args.mode == "test":
		test()

