'''
this file defines our main networks
'''

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORM = False


class Flatten(nn.Module):
	'''nn.Module that flattens the input'''
	def forward(self, x:torch.Tensor):
		if len(x.shape) > 3:
			N, C, H, W = x.size()
			return x.view(N, -1)
		else:
			return x.view(-1)

class ConcatHelper(nn.Module):
	'''this class concatenates two inputs, so it can be fed into an nn.Sequential'''
	def __init__(self, network:nn.Module, prenetwork:nn.Module = None):
		'''
		args:
		- `network`: a network where the concatenated input will be passed into
		'''
		super().__init__()
		self.network = network
		self.prenetwork = prenetwork

	def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
		y_ndim = y.ndim
		
		if self.prenetwork:
			x = self.prenetwork(x)
		if x.shape[0] != y.shape[0] and y_ndim > 1:
			y = torch.mean(y.clone(), dim=0, keepdim=True)
			y = y.repeat(x.clone().shape[0], 1)
		y = y.unsqueeze(-1).unsqueeze(-1).expand(y.shape + x.shape[-2:])
		out = torch.cat((x, y), dim=(y_ndim - 1))
		out = self.network(out)
		return out

class DownsampleBertonBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
		self.relu = nn.LeakyReLU(0.01, inplace=True)
	
	def forward(self, x):
		out = self.conv(x)
		out = self.relu(out)
		return out

class UpsampleBertonBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, (4, 4), 2, padding=1)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.conv_t(x)
		out = self.relu(out)
		return out


class ResidualBertonBlock(nn.Module):
	def __init__(self, channels, leaky=True, norm=False):
		super().__init__()
		self.has_norm = norm
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
		self.relu = nn.LeakyReLU(0.01, inplace=True) if leaky else nn.ReLU(inplace=True)
		if norm:
			self.norm1 = nn.InstanceNorm2d(channels, momentum=1)
			self.norm2 = nn.InstanceNorm2d(channels, momentum=1)

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		# if self.has_norm:
		# 	out = self.norm1(out)
		out = self.relu(out)
		out = self.conv2(out)
		if self.has_norm:
			out = self.norm2(out)
		out = out + identity
		out = self.relu(out)
		return out

# input: n x 3 x 128 x 128
# output: n x 32
_celeba_face_encoder:nn.Module = nn.Sequential(
	# nn.InstanceNorm2d(3, momentum=1, affine=True),
	DownsampleBertonBlock(3, 32),
	DownsampleBertonBlock(32, 64), # (64 x 32 x 32)
	ResidualBertonBlock(64),
	DownsampleBertonBlock(64, 64), # (64 x 16 x 16)
	DownsampleBertonBlock(64, 128), # (128 x 8 x 8)
	DownsampleBertonBlock(128, 128),
	DownsampleBertonBlock(128, 256), # (128 x 2 x 2)
	nn.MaxPool2d(kernel_size=2), # (128,)
	Flatten(),
	nn.Linear(256, 64),
	nn.LeakyReLU(0.01),
	nn.Linear(64, 32),
)

# input: N x 3 x 128 x 128
# output: N x 32 x 8 x 8
_celeba_image_encoder:nn.Module = nn.Sequential(
	# nn.InstanceNorm2d(3, momentum=1, affine=True),
	DownsampleBertonBlock(3, 32),
	DownsampleBertonBlock(32, 64), # (64 x 32 x 32)
	ResidualBertonBlock(64),
	DownsampleBertonBlock(64, 64), # (64 x 16 x 16)
	DownsampleBertonBlock(64, 128), # (128 x 8 x 8)
	ResidualBertonBlock(128),
	nn.Conv2d(128, 32, kernel_size=1),
)

# input: N x 64 x 8 x 8
# output: N x 3 x 128 x 128
_celeba_image_decoder:nn.Module = ConcatHelper(nn.Sequential(
	UpsampleBertonBlock(64, 64),
	ResidualBertonBlock(64, leaky=False),
	UpsampleBertonBlock(64, 32), # (32 x 32 x 32)
	UpsampleBertonBlock(32, 32), # (32 x 64 x 64)
	ResidualBertonBlock(32, leaky=False),
	UpsampleBertonBlock(32, 16), # (16 x 128 x 128)
	nn.Conv2d(16, 3, kernel_size=1),
	nn.Sigmoid()
))

# input: N x 3 x 128 x 128
# output: N x 1
_celeba_discriminator1:nn.Module = nn.Sequential(
	# nn.InstanceNorm2d(3, affine=True),
	DownsampleBertonBlock(3, 32),
	DownsampleBertonBlock(32, 64), # (64 x 32 x 32)
	ResidualBertonBlock(64),
	DownsampleBertonBlock(64, 64), # (64 x 16 x 16)
	DownsampleBertonBlock(64, 128), # (128 x 8 x 8)
	DownsampleBertonBlock(128, 128),
	DownsampleBertonBlock(128, 256), # (128 x 2 x 2)
	nn.MaxPool2d(kernel_size=2), # (128,)
	Flatten(),
	nn.Linear(256, 64),
	nn.LeakyReLU(0.01),
	nn.Linear(64, 16),
	nn.LeakyReLU(0.01),
	nn.Linear(16, 1)
)

# input: x: (N x 3 x 128 x 128) y: (n x 16)
# output: N x 1
_celeba_discriminator2:nn.Module = ConcatHelper(
	prenetwork=nn.Sequential(
		# nn.InstanceNorm2d(3, affine=True),
		DownsampleBertonBlock(3, 32),
		DownsampleBertonBlock(32, 64),
		DownsampleBertonBlock(64, 32),
		nn.LeakyReLU(0.01)  # now we have 32 x 16 x 16
	),
	network=nn.Sequential(
		DownsampleBertonBlock(64, 64), # (64 x 16 x 16)
		DownsampleBertonBlock(64, 128), # (64 x 8 x 8),
		DownsampleBertonBlock(128, 256), # (128 x 4 x 4)
		nn.MaxPool2d(kernel_size=2),
		Flatten(),
		nn.Linear(256, 128),
		nn.LeakyReLU(),
		nn.Linear(128, 32),
		nn.LeakyReLU(),
		nn.Linear(32, 1)
	)
)


class CelebBlock(nn.Module):
	def __init__(self, Cin, Cout):
		super().__init__()

		self.net = nn.Sequential(
			nn.Conv2d(Cin, int(Cin/4), 1, 1),
			nn.LeakyReLU(0.01),
			nn.Conv2d(int(Cin/4), int(Cin/2), 3, 1, 1),
			nn.LeakyReLU(0.01),
			nn.Conv2d(int(Cin/2), int(Cin/2), 5, 1, 2),
			nn.LeakyReLU(0.01), 
			nn.Conv2d(int(Cin/2), Cout, 1, 1)
		)

class InitialCelebBlock(nn.Module):
	def __init__(self, Cout):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(1, Cout, 1, 1)
		)

class LastCelebBlock(nn.Module):
	def __init__(self, Cin):
		super().__init__()
		self.net = nn.Sequential(
			Flatten(),
			nn.Linear(1024, 4*4*64),
			nn.LeakyReLU(0.01),
			nn.Linear(4*4*64, 2)
		)




networks = {
	'mnist': {
		'face_encoder': nn.Sequential(
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
		), # nn.Sequential or something: CNN: 28x28x1 --> 2
		'image_encoder': nn.Sequential(
			nn.Conv2d(1, 8, 3, 1, 1),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(8, 16, 3, 1, 1),
			nn.LeakyReLU(0.01),
			nn.Conv2d(16, 32, 3, 1, 1),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d(2,2)
	), # FCNN: 28x28x1 --> some feature map (maybe 7x7x8)
		'image_decoder': ConcatHelper(nn.Sequential(
			nn.Conv2d(34, 128, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(128, 32, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 16, (4, 4), 2, padding=1),
			nn.ReLU(),
			nn.Conv2d(16, 16, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.ConvTranspose2d(16, 1, (4, 4), 2, padding=1),
			# nn.Tanh(),
			nn.Sigmoid(),
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
			nn.Linear(64 * 4 * 4, 64 * 4 * 4),
			nn.LeakyReLU(0.01),
			nn.Linear(64 * 4 * 4, 10),
			nn.LeakyReLU(0.01),
			nn.Linear(10, 1),
			# nn.Sigmoid(),
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
	'celeba': {
		'face_encoder': _celeba_face_encoder,
		'image_encoder': _celeba_image_encoder,
		'image_decoder': _celeba_image_decoder,
		'discriminator1': _celeba_discriminator1,
		'discriminator2': _celeba_discriminator2,
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
	def __init__(self, type:str='mnist', device=DEVICE):
		self.type = type
		self.face_encoder:nn.Module = networks[type]['face_encoder']
		self.image_encoder:nn.Module = networks[type]['image_encoder']
		self.image_decoder:nn.Module = networks[type]['image_decoder']
		self.discriminator1:nn.Module = networks[type]['discriminator1']
		self.discriminator2:nn.Module = networks[type]['discriminator2']
		if type != 'empty':
			self.to(device)
	
	def to(self, device):
		self.face_encoder.to(device)
		self.image_encoder.to(device)
		self.image_decoder.to(device)
		self.discriminator1.to(device)
		self.discriminator2.to(device)

	def eval(self):
		self.face_encoder.eval()
		self.image_encoder.eval()
		self.image_decoder.eval()
		self.discriminator1.eval()
		self.discriminator2.eval()
	
	def train(self):
		self.face_encoder.train()
		self.image_encoder.train()
		self.image_decoder.train()
		self.discriminator1.train()
		self.discriminator2.train()

	def shares_style(self, F_A:torch.Tensor, I:torch.Tensor):
		'''
		input: a set of images with style A, and image with unknown style. 
		output: probability of unknown style being same as A
		'''
		self.eval()
		with torch.no_grad():
			h_A = self.face_encoder(F_A)
			probit = self.discriminator2(I, h_A)
		return probit

	def shares_style_latent(self, I:torch.Tensor, h_A:torch.Tensor) -> torch.Tensor:
		self.eval()
		with torch.no_grad():
			probit = self.discriminator2(I, h_A)
		return probit

	def generate_image(self, F_A, I):
		'''
		input: a set of images with style A, and other image, I
		output: new image similar to I, with style A
		'''
		self.eval()
		with torch.no_grad():
			h_A = self.face_encoder(F_A)
			h_I = self.image_encoder(I)
			I_fake = self.image_decoder(h_I, h_A)
		return I_fake
		

	def generate_image_latent(self, I, h_A):
		self.eval()
		with torch.no_grad():
			h_I = self.image_encoder(I)
			I_fake = self.image_decoder(h_I, h_A)
		return I_fake



class Unflatten(nn.Module):
  """
  An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
  to produce an output of shape (N, C, H, W).
  """
  def __init__(self, N=-1, C=128, H=7, W=7):
    super(Unflatten, self).__init__()
    self.N = N
    self.C = C
    self.H = H
    self.W = W
  def forward(self, x):
    return x.view(self.N, self.C, self.H, self.W)
