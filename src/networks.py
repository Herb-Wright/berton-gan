'''
this file defines our main networks
'''

import torch
import torch.nn as nn




networks = {
	'mnist': {
		'face_encoder': NotImplemented, # nn.Sequential or something
		'image_encoder': NotImplemented,
		'image_decoder': NotImplemented,
		'discriminator': NotImplemented,
	}
}

class BertonGan():
	def __init__(self, type='mnist'):
		self.type = type
		self.face_encoder = networks[type]['face_encoder']
		self.image_encoder = networks[type]['image_encoder']
		self.image_decoder = networks[type]['image_decoder']
		self.discriminator = networks[type]['discriminator']
		
	
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


