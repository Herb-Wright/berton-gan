'''
This file contains utility functions for running our experiments,
including saving and loading models and checkpoints, and dealing
with files
'''

import sys
import os
import torch
from typing import Tuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import BertonGan

MODELS_PATH = os.path.abspath(os.path.join(
	os.path.dirname(__file__),
	'../models'
))

def model_path_exists(name, path=MODELS_PATH):
	return os.path.exists(os.path.join(MODELS_PATH, name))

def create_dir_if_not_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)

def save_berton_gan(gan:BertonGan, name:str, path:str=MODELS_PATH, type:str='torchscript'):
	dir = os.path.join(path, name)
	create_dir_if_not_exists(dir)
	if type == 'torchscript':
		torch.jit.script(gan.face_encoder).save(os.path.join(dir, 'face_encoder.pt'))
		torch.jit.script(gan.image_encoder).save(os.path.join(dir, 'image_encoder.pt'))
		torch.jit.script(gan.image_decoder).save(os.path.join(dir, 'image_decoder.pt'))
		torch.jit.script(gan.discriminator1).save(os.path.join(dir, 'discriminator1.pt'))
		torch.jit.script(gan.discriminator2).save(os.path.join(dir, 'discriminator2.pt'))
	else:
		raise Exception('invalid saving type')


def load_berton_gan(name:str, path:str=MODELS_PATH) -> BertonGan:
	dir = os.path.join(path, name)
	berton_gan = BertonGan('empty')
	berton_gan.face_encoder = torch.jit.load(os.path.join(dir, 'face_encoder.pt')).to(DEVICE)
	berton_gan.image_encoder = torch.jit.load(os.path.join(dir, 'image_encoder.pt')).to(DEVICE)
	berton_gan.image_decoder = torch.jit.load(os.path.join(dir, 'image_decoder.pt')).to(DEVICE)
	berton_gan.discriminator1 = torch.jit.load(os.path.join(dir, 'discriminator1.pt')).to(DEVICE)
	berton_gan.discriminator2 = torch.jit.load(os.path.join(dir, 'discriminator2.pt')).to(DEVICE)
	return berton_gan

def save_checkpoint(gan:BertonGan, checkpoint:dict, checkpoint_name:str, path:str=MODELS_PATH):
	'''
	saves a checkpoint

	args:
	- `gan`: BertonGan to be saved
	- `checkpoint`: dict with values to be saved for checkpoint
	- `checkpoint_name`: the name of the checkpoint
	- `path`: path to the models directory

	this function returns nothing
	'''
	save_berton_gan(gan, checkpoint_name, path)
	dir = os.path.join(path, checkpoint_name)
	create_dir_if_not_exists(dir)
	torch.save(checkpoint, os.path.join(dir, 'checkpoint.pt'))

def load_checkpoint(checkpoint_name:str, path:str=MODELS_PATH) -> Tuple[BertonGan, dict]:
	'''
	loads a checkpoint

	args:
	- `checkpoint_name`: name of checkpoint
	- `path`: path to models directory

	returns:
	- berton_gan at checkpoint
	- checkpoint dict
	'''
	berton_gan = load_berton_gan(checkpoint_name, path)
	dir = os.path.join(path, checkpoint_name)
	checkpoint_dict = torch.load(os.path.join(dir, 'checkpoint.pt'))
	return berton_gan, checkpoint_dict
