'''
this file defines our main networks
'''

import torch
import torch.nn as nn




networks = {
	'mnist': {
		'face_encoder': NotImplemented, # nn.Sequential or something: CNN: 28x28x1 --> 2
		'image_encoder': NotImplemented, # FCNN: 28x28x1 --> some feature map (maybe 7x7x8)
		'image_decoder': NotImplemented, # FCNN: (feature map) x 2 --> 28x28x1
		'discriminator1': NotImplemented, # CNN: (28x28x1) --> 1
		'discriminator2': NotImplemented, # CNN: (28x28x1) x 2	--> 2
	}
}

class BertonGan():
	def __init__(self, type='mnist'):
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

	def shares_style(self, F_A, I):
		'''
		input: a set of images with style A, and image with unknown style. 
		output: probability of unknown style being same as A
		'''
		pass

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
	
class Flatten(nn.Module):
	def forward(self, x):
		N, C, H, W = x.size()
		return x.view(N, -1)


