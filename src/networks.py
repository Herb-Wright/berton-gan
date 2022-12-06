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


