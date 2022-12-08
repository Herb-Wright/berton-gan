
import math
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

def load_mnist_data():
	NOISE_DIM = 96
	batch_size = 128

	print('download MNIST if not exist')

	mnist_train = dset.MNIST('./data/MNIST_data', train=True, download=True,
                           	transform=T.ToTensor())	
	loader_train = DataLoader(mnist_train, batch_size=batch_size,
                          	shuffle=True, drop_last=True, num_workers=2)


def load_msceleb():
	pass



