
import torch
from torch.optim import SGD 
from torch.nn.functional import mse_loss

def _get_optimizer_options(optimizer_options):
	'''
	if a learning rate is not specified, it is set to `1e-4`,
	then updated `optimizer_options` dict is returned
	'''
	for optim_name in ['face_encoder', 'image_decoder', 'discriminator']:
		if optim_name not in optimizer_options.keys():
			optimizer_options[optim_name] = {'lr': 1e-4}
	return optimizer_options

# square losses for the network
_square_losses = {
	'face_encoder': lambda C_A, C_A_fake, C_B: 
		(mse_loss(C_A, torch.ones_like(C_A)) + mse_loss(C_A_fake, torch.ones_like(C_A_fake)) + mse_loss(C_B, torch.zeros_like(C_B))) / 3,
	'image_decoder': lambda R_A_fake, R_B_fake, C_A_fake, C_B_fake, D_A, D_B: 
		(mse_loss(R_A_fake, torch.ones_like(R_A_fake)) 
			+ mse_loss(R_B_fake, torch.ones_like(R_B_fake)) 
			+ mse_loss(C_A_fake, torch.ones_like(C_A_fake)) 
			+ mse_loss(C_B_fake, torch.ones_like(C_B_fake)) 
			+ D_A + D_B) / 6,
	'discriminator': lambda R_A, R_B, C_A, R_A_fake, R_B_fake, C_B:
		(mse_loss(R_A, torch.ones_like(R_A)) 
			+ mse_loss(R_B, torch.ones_like(R_B)) 
			+ 2 * mse_loss(C_A, torch.ones_like(C_A)) 
			+ mse_loss(R_A_fake, torch.zeros_like(R_A_fake)) 
			+ mse_loss(R_B_fake, torch.zeros_like(R_B_fake)) 
			+ 2 * mse_loss(C_B, torch.zeros_like(C_B))) / 4,
}

def _forward_fake(gan, data, h_F=None):
	'''
	computes forward pass on network from images to discriminator (fake)
	input
		- gan: a BertonGan instance
		- data: a batch from dataloader
	output:
		- I_A_fake, I_B_fake, R_A_fake, R_B_fake, C_A_fake, C_B_fake quantities
	'''
	f_A, I_A, I_B = data
	if h_F is None:
		h_F = gan.face_encoder(f_A)
	h_I = gan.image_encoder(I_A)
	h_B = gan.image_encoder(I_B)
	I_A_fake = gan.image_decoder(h_I, h_F)
	I_B_fake = gan.image_decoder(h_B, h_F)
	R_A_fake = gan.discriminator1(I_A_fake)
	R_B_fake = gan.discriminator1(I_B_fake)
	C_A_fake = gan.discriminator2(I_A_fake, h_F)
	C_B_fake = gan.discriminator2(I_B_fake, h_F)
	return I_A_fake, I_B_fake, R_A_fake, R_B_fake, C_A_fake, C_B_fake

def _forward_real(gan, data, h_F=None):
	'''
	computes discriminator output for real images
	input: 
		- gan: a BertonGan instance
		- data: batch of (f_A, I_A, I_B)
	output: R_A, R_B, C_A, C_B quantities
	'''
	f_A, I_A, I_B = data
	if h_F is None:
		h_F = gan.face_encoder(f_A)
	R_A = gan.discriminator1(I_A)
	R_B = gan.discriminator1(I_B)
	C_A = gan.discriminator2(I_A, h_F)
	C_B = gan.discriminator2(I_B, h_F)
	return R_A, R_B, C_A, C_B
	
def _train_one_epoch(
	gan, 
	dataloader, 
	losses=_square_losses,
	dist_func=(lambda x, y: torch.square(x - y).sum()),
	F_optim=None, 
	G_optim=None, 
	D_optim=None,
):
	'''
	function that trains a network for one epoch.
	If F_optim, G_optim, or D_optim are `None`, then that network is skipped for backprop.
	'''
	F_loss_total, G_loss_total, D_loss_total = 0, 0, 0
	for i, data in enumerate(dataloader):
		f_A, I_A, I_B = data
		h_F = gan.face_encoder(f_A)
		# backprop on the face encoder
		if F_optim:
			F_optim.zero_grad()
			I_A_fake, I_B_fake, R_A_fake, R_B_fake, C_A_fake, C_B_fake = _forward_fake(gan, data, h_F)
			R_A, R_B, C_A, C_B = _forward_real(gan, data, h_F)
			F_loss = losses['face_encoder'](C_A, C_A_fake, C_B)
			F_loss.backward()
			F_optim.step()
			F_loss_total += F_loss
		h_F = h_F.detach()
		# backprop on the image encoder and decoder
		if G_optim:
			G_optim.zero_grad()
			I_A_fake, I_B_fake, R_A_fake, R_B_fake, C_A_fake, C_B_fake = _forward_fake(gan, data, h_F)
			D_A = dist_func(I_A_fake, I_A)
			D_B = dist_func(I_B_fake, I_B)
			G_loss = losses['image_decoder'](R_A_fake, R_B_fake, C_A_fake, C_B_fake, D_A, D_B)
			G_loss
			G_loss.backward()
			G_optim.step()
			G_loss_total += G_loss
		# backprop on the discriminators
		if D_optim:
			D_optim.zero_grad()
			if F_optim or G_optim:
				I_A_fake = I_A_fake.detach()
				I_B_fake = I_B_fake.detach()
				R_A_fake = gan.discriminator1(I_A_fake)
				R_B_fake = gan.discriminator1(I_B_fake)
			else:
				I_A_fake, I_B_fake, R_A_fake, R_B_fake, C_A_fake, C_B_fake = _forward_fake(gan, data, h_F)
			R_A, R_B, C_A, C_B = _forward_real(gan, data, h_F)
			D_loss = losses['discriminator'](R_A, R_B, C_A, R_A_fake, R_B_fake, C_B)
			D_loss.backward()
			D_optim.step()
			D_loss_total += D_loss
	# return all our losses
	return F_loss_total, G_loss_total, D_loss_total	

def train_all_at_once(
	berton_gan, 
	dataloader, 
	epochs=40,
	optimizer=SGD, 
	optimizer_options={},
	verbose=False,
	evaluator=None
):
	'''
	trains the gan by training each network simultaneously (possibly less stable)
	inputs:
	- berton_gan: a BertonGan instance
	- dataloader: a dataloader specific for BertonGan training
	- epochs: integer number of epochs (optional)
	- optimizer: a class of optimizer (i.e. `torch.optim.SGD`)
	- verbose: boolean that specifies whether to print at each epoch
	- evaluator: a function that takes in a berton_gan and outputs a number,
		will be run on the berton_gan at each iteration, to evaluate it.
	returns:
	- dict of epoch training (keys are integer of epoch 0, ..., epochs - 1)
		and values are another dict with losses and the value of an optional
		evaluator function (if specified)
	'''
	# construct our optimizers
	optimizer_options = _get_optimizer_options(optimizer_options)
	F_optim = optimizer(berton_gan.face_encoder.parameters(), **optimizer_options['face_encoder'])
	G_optim = optimizer(
		list(berton_gan.image_encoder.parameters()) + list(berton_gan.image_decoder.parameters()),
		**optimizer_options['image_decoder'],
	)
	D_optim = optimizer(
		list(berton_gan.discriminator1.parameters()) + list(berton_gan.discriminator2.parameters()),
		**optimizer_options['discriminator'],
	)
	metadata = {}
	# train the network
	for epoch in range(epochs):
		if verbose:
			print(f'epoch {epoch}')
		# do the epoch
		F_loss, G_loss, D_loss = _train_one_epoch(
			berton_gan, 
			dataloader,
			F_optim=F_optim,
			G_optim=G_optim,
			D_optim=D_optim,
		)
		# log data
		metadata[epoch]  = {
			'F_loss': F_loss,
			'G_loss': G_loss,
			'D_loss': D_loss,
		}
		if evaluator:
			with torch.no_grad:
				eval = evaluator(berton_gan)
			metadata[epoch]['eval'] = eval
		# print stuff
		if verbose:
			print(f'  F_loss: {F_loss}; G_loss: {G_loss}; D_loss: {D_loss}')
			if evaluator:
				print(f'  Evaluation: {eval}')
	return metadata




def train_rotate(
	network, 
	dataloader, 
	epoch_per=2,
	epochs=40,
	optimizer=SGD, 
	optimizer_options={}, 
	verbose=False
):
	'''
	trains the gan one network at a time (maybe more stable)
	returns:
		- data on training
	'''
	raise NotImplementedError


# Run tests
if __name__ == '__main__':
	print('running tests for src/training.py')
	n, N, I, H_f, H_i = 3, 5, 8, 4, 6
	random_dataloader = [(torch.rand(n, I), torch.rand(N, I), torch.rand(N, I)) for i in range(3)]
	import sys
	print(sys.path)
	torch.autograd.set_detect_anomaly(True)
	from networks import BertonGan
	import torch.nn as nn
	class ConcatHelper(nn.Module):
		def __init__(self, network):
			super().__init__()
			self.network = network
		def forward(self, x, y):
			if x.shape[0] != y.shape[0]:
				y, _ = torch.max(y.clone(), dim=0, keepdim=True)
				y = y.repeat(x.clone().shape[0], 1)
			out = torch.cat((x, y), dim=1)
			out = self.network(out)
			return out
	random_berton_gan = BertonGan()
	random_berton_gan.face_encoder = nn.Linear(I, H_f)
	random_berton_gan.image_encoder = nn.Linear(I, H_i)
	random_berton_gan.image_decoder = ConcatHelper(nn.Linear(H_f + H_i, I))
	random_berton_gan.discriminator1 = nn.Linear(I, 1)
	random_berton_gan.discriminator2 = ConcatHelper(nn.Linear(I + H_f, 1))
	
	# _forward_fake tests
	print('  TEST _forward_fake:')
	I_A_fake, I_B_fake, R_A_fake, R_B_fake, C_A_fake, C_B_fake = _forward_fake(random_berton_gan, random_dataloader[0])
	assert(I_A_fake.shape == (N, I))
	assert(I_B_fake.shape == (N, I))
	print('    fake images have correct shape')
	assert(R_A_fake.shape == (N, 1))
	assert(R_B_fake.shape == (N, 1))
	assert(C_A_fake.shape == (N, 1))
	assert(C_B_fake.shape == (N, 1))
	print('    discriminator outputs have correct shape')
	print('  ...PASSED')

	# train_all_at_once tests
	print('  TEST train_all_at_once:')
	B = 5
	data = train_all_at_once(random_berton_gan, random_dataloader, B)
	print('    function ran with no errors')
	assert(data[0]['F_loss'] >= data[B - 1]['F_loss'])
	print('    F loss function was improved')
	assert(data[0]['G_loss'] >= data[B - 1]['G_loss'])
	print('    G loss function was improved')
	assert(data[0]['D_loss'] >= data[B - 1]['D_loss'])
	print('    D loss function was improved')
	print('  ...PASSED')






