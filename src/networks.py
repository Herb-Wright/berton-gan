'''
this file defines our main networks
'''

import torch
import torch.nn as nn


class ConcatHelper(nn.Module):
		def __init__(self, network):
			super().__init__()
			self.network = network

		def forward(self, x:torch.Tensor, y:torch.Tensor):
			y_ndim = y.ndim
			if x.shape[0] != y.shape[0] and y_ndim > 1:
				y = torch.mean(y.clone(), dim=0, keepdim=True)
				y = y.repeat(x.clone().shape[0], 1)
			y = y.unsqueeze(-1).unsqueeze(-1).expand(y.shape + x.shape[-2:])
			out = torch.cat((x, y), dim=(y_ndim - 1))
			out = self.network(out)
			return out


class Flatten(nn.Module):
	'''nn.Module that flattens the input'''
	def forward(self, x:torch.Tensor):
		if len(x.shape) > 3:
			N, C, H, W = x.size()
			return x.view(N, -1)
		else:
			return x.view(-1)

def _mnist_face_encoder():
	'''
	This is a face encoder network using nn.sequential

	The layers are as follows:
	1. Convolutional layer with 32 filters, 5x5 size, stride 1
	2. Leaky ReLU
	3. 2x2 Max pooling layer, stride 2
	4. Convolutional layer with 64 filters, 5x5 size, stride 1
	5. Leaky ReLu
	6. 2x2 Max pooling layer, stride 2
	7. Flattening the matrix before the fully connected layer
	8. Fully connected layer with output size 4x4x64
	9. Leaky ReLU
	10. Fully connected layer with output size of 2
	'''
	return nn.Sequential(
			nn.Conv2d(1,32,5,1),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d(2,2),
			nn.Conv2d(32,64,5,1),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d(2,2),
			Flatten(),
			nn.Linear(1024,4*4*64),
			nn.LeakyReLU(0.01),
			nn.Linear(4*4*64,2)
		)

def _mnist_image_encoder():
	'''
	1. Convolutional layer with 8 filters, 3x3 size, padding 1, stride 1
	2. ReLU
	3. 2x2 Max pooling layer stride 2
	4. Convolutional layer with 16 filters 3x3 size, padding 1, stride 1
	5. ReLU
	6. Convolutional layer with 32 filters 3x3 size, padding 1, stride 1
	7. ReLU
	8. 2x2 Max pooling layer stride 2

	Output is a 7x7x32 tensor
	'''
	return nn.Sequential(
		nn.Conv2d(1, 8, 3, 1, 1),
		nn.ReLU(),
		nn.MaxPool2d(2, 2),
		nn.Conv2d(8, 16, 3, 1, 1),
		nn.ReLU(),
		nn.Conv2d(16, 32, 3, 1, 1),
		nn.ReLU(),
		nn.MaxPool2d(2,2)
	)


networks = {
	'mnist': {
		'face_encoder': _mnist_face_encoder(), # nn.Sequential or something: CNN: 28x28x1 --> 2
		'image_encoder': _mnist_image_encoder(), # FCNN: 28x28x1 --> some feature map (maybe 7x7x8)
		'image_decoder': ConcatHelper(nn.Sequential(
			nn.Conv2d(34, 64, kernel_size=3, padding='same'),
			nn.LeakyReLU(0.01),
			nn.Conv2d(64, 32, kernel_size=3, padding='same'),
			nn.LeakyReLU(0.01),
			nn.ConvTranspose2d(32, 16, (4, 4), 2, padding=1),
			nn.LeakyReLU(0.01),
			nn.Conv2d(16, 16, kernel_size=3, padding='same'),
			nn.LeakyReLU(0.01),
			nn.ConvTranspose2d(16, 1, (4, 4), 2, padding=1),
			nn.Tanh(),
		)), # FCNN: (feature map) x 2 --> 28x28x1
		'discriminator1': nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(5, 5)),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d((2, 2)),
			nn.Conv2d(32, 64, kernel_size=(5, 5)),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d((2, 2)),
			Flatten(),
			nn.Linear(64 * 4 * 4, 64 * 4 * 4),
			nn.LeakyReLU(0.01),
			nn.Linear(64 * 4 * 4, 1),
			nn.Sigmoid(),
		), # CNN: (28x28x1) --> 1
		'discriminator2': ConcatHelper(nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=(5, 5)),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d((2, 2)),
			nn.Conv2d(32, 64, kernel_size=(5, 5)),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d((2, 2)),
			Flatten(),
			nn.Linear(64 * 4 * 4, 64 * 4 * 4),
			nn.LeakyReLU(0.01),
			nn.Linear(64 * 4 * 4, 1),
			nn.Sigmoid(),
		)), # CNN: (28x28x1) x 2	--> 1
	},
	'empty': {
		'face_encoder': None,
		'image_encoder': None,
		'image_decoder': None,
		'discriminator1': None,
		'discriminator2': None,
	},
}

class BertonGan():
	def __init__(self, type:str='mnist'):
		self.type = type
		self.face_encoder = networks[type]['face_encoder']
		self.image_encoder = networks[type]['image_encoder']
		self.image_decoder = networks[type]['image_decoder']
		self.discriminator1 = networks[type]['discriminator1']
		self.discriminator2 = networks[type]['discriminator2']
		
	
	def load_weights(self):
		'''
		loads pretrained weights of the model
		'''
		pass

	def shares_style(self, F_A:torch.Tensor, I:torch.Tensor):
		'''
		input: a set of images with style A, and image with unknown style. 
		output: probability of unknown style being same as A
		'''
		pass

	def shares_style_latent(self, I:torch.Tensor, h_A:torch.Tensor) -> torch.Tensor:
		return self.discriminator2(I, h_A)

	def generate_image(self, F_A, I):
		'''
		input: a set of images with style A, and other image, I
		output: new image similar to I, with style A
		'''
		pass

	def face_encoder():
		'''
		This is a face encoder network using nn.sequential

		The layers are as follows:
		1. Convolutional layer with 32 filters, 5x5 size, stride 1
		2. Leaky ReLU
		3. 2x2 Max pooling layer, stride 2
		4. Convolutional layer with 64 filters, 5x5 size, stride 1
		5. Leaky ReLu
		6. 2x2 Max pooling layer, stride 2
		7. Flattening the matrix before the fully connected layer
		8. Fully connected layer with output size 4x4x64
		9. Leaky ReLU
		10. Fully connected layer with output size of 2
		'''
		return nn.Sequential(
    		nn.Conv2d(1,32,5,1),
    		nn.LeakyReLU(0.01),
    		nn.MaxPool2d(2,2),
    		nn.Conv2d(32,64,5,1),
    		nn.LeakyReLU(0.01),
    		nn.MaxPool2d(2,2),
    		Flatten(),
    		nn.Linear(1024,4*4*64),
    		nn.LeakyReLU(0.01),
    		nn.Linear(4*4*64,2)
  		)
	
	def image_encoder():
		'''
		1. Convolutional layer with 8 filters, 3x3 size, padding 1, stride 1
		2. ReLU
		3. 2x2 Max pooling layer stride 2
		4. Convolutional layer with 16 filters 3x3 size, padding 1, stride 1
		5. ReLU
		6. Convolutional layer with 32 filters 3x3 size, padding 1, stride 1
		7. ReLU
		8. 2x2 Max pooling layer stride 2

		Output is a 7x7x32 tensor
		'''
		return nn.Sequential(
			nn.Conv2d(1, 8, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(8, 16, 3, 1, 1),
			nn.ReLU(),
			nn.Conv2d(16, 32, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(2,2)
		)

	



