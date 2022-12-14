
import torch
import torchvision.transforms as T
import torchvision.datasets as dset
import os

# CONSTANTS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_PATH = os.path.abspath(
	os.path.join(
		os.path.abspath(__file__), 
		'../../data'
	)
)

# FOR DOWNLOADING DATA

def download_mnist_data(path=DATA_PATH, train=True):
	mnist_train = dset.MNIST(path, train=train, download=True, transform=T.ToTensor())
	return mnist_train

# NOTE: idk if we want the msceleb-1m dataset anymore, it has been retracted:
# https://paperswithcode.com/dataset/ms-celeb-1m
# but maybe we could use one of these instead:
# https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
# http://umdfaces.io/
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html <-- this one is another celeb dataset
# https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
def download_msceleb(path=DATA_PATH, train=True):
	msceleb_train = dset.CelebA(path, 'train', download=True, transform=T.ToTensor())
	return msceleb_train

# download_msceleb(DATA_PATH, True)


# DATA LOADERS FOR CONSTRUCTING BATCHES

class MnistLoader:
	def __init__(
		self, 
		encoder_amount=3, 
		batch_size=32, 
		transform=None, 
		path=DATA_PATH, 
		train=True, 
		device=DEVICE
	):
		'''
		Creates a loader for MNIST data.

		args: 
		- `encoder_amount` (int): number of examples in f_A (|f_A| = n)
		- `batch_size` (int): number of examples in I_A, I_B (|I_A| = |I_B| = N)
		- `transform` (torchvision.Transform): transform for data at each batch
		- `path` (str): a path to where the data directory is
		- `train` (bool): whether or not to load the training data
		- `device` (torch.device): device to host data in batch

		returns:
		- (MnistLoader): a loader that loads an epoch of MNIST
		'''
		self.data_length = 50000 if train else 10000
		self.batch_size = batch_size
		self.encoder_amount = encoder_amount
		self.device = device
		self.transform = T.Compose(
			[T.PILToTensor(), transform] if transform else [T.PILToTensor()]
		)
		self.data_set = dset.MNIST(path, train=train)

	def __iter__(self):
		'''starts MnistLoader iterator'''
		self.count = 0
		return self

	def __next__(self):
		'''
		randomly generates indices, then iteratively builds a batch 
		
			B = (f_A, I_A, I_B) 
		
		and returns it

		returns:
		- (tuple): a batch B s.t. f_A is (n, 1, 28, 28) and I_A, I_B are (N, 1, 28, 28)
		'''
		if self.count >= self.data_length:
			raise StopIteration
		I_A, I_B = [], []
		random_index = torch.randint(self.data_length, (1,))[0]
		x_A, A_label = self.data_set[random_index]
		F_A = [self.transform(x_A)]
		while (len(F_A) < self.encoder_amount or len(I_A) < self.batch_size or len(I_B) < self.batch_size): 
			random_indices = torch.randint(self.data_length, (100,))
			for i in random_indices:
				x_i, y_i = self.data_set[i]
				if y_i == A_label:
					if len(I_A) < self.batch_size:
						I_A.append(self.transform(x_i))
					elif len(F_A) < self.encoder_amount:
						F_A.append(self.transform(x_i))
				elif len(I_B) < self.batch_size:
					I_B.append(self.transform(x_i))
		self.count += 2 * self.batch_size + self.encoder_amount
		F_A = torch.stack(F_A).to(self.device)
		I_A = torch.stack(I_A).to(self.device)
		I_B = torch.stack(I_B).to(self.device)
		return (F_A, I_A, I_B)



if __name__ == '__main__':

	# test loading in data
	print('  TEST download_mnist_data')
	download_mnist_data()
	print('    function runs without error')
	print('  ...PASSED')

	# test MnistLoader
	print('  TEST MnistLoader')
	n, N = 3, 16
	dataloader = MnistLoader(encoder_amount=n, batch_size=N)
	for i, data in enumerate(dataloader):
		f_A, I_A, I_B = data
		assert(f_A.shape == (n, 1, 28, 28))
		assert(I_A.shape == (N, 1, 28, 28))
		assert(I_B.shape == (N, 1, 28, 28))
		if i >= 5:
			break
	print('    MnistLoader returns batch with correct shape')
	import time
	start = time.time()
	my_num = 0
	for i, data in enumerate(dataloader):
		my_num += 1 # do a small amount of work
		my_num -= 1
	end = time.time()
	assert(end - start < 8)
	print(f'    One epoch completed in {end - start} seconds')
	print('  ...PASSED')

