
import torch
import torchvision.transforms as T
import torchvision.datasets as dset
from torchvision.datasets.vision import VisionDataset
import os
import math
import requests
import zipfile
from typing import Any, Callable, List, Optional, Tuple, Union
from torchvision.datasets.utils import verify_str_arg
import csv
import PIL
from collections import namedtuple
from tqdm import trange


# CONSTANTS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_PATH = os.path.abspath(
	os.path.join(
		os.path.abspath(__file__), 
		'../../data'
	)
)

# FOR DOWNLOADING DATA

def download_mnist_data(path:str=DATA_PATH, train:str=True):
	'''
	downloads mnist data into the `path` directory.
	Then returns the pytorch dataset

	args:
	- `path`: the path to download MNIST to
	- `train`: whether or not to return the training data or test data

	returns:
	- the dataset
	'''
	mnist_train = dset.MNIST(path, train=train, download=True, transform=T.ToTensor())
	return mnist_train

def download_celeba(path=DATA_PATH, train=True):
	if not os.path.exists(path):
		os.makedirs(path)
	dataset_folder = os.path.join(path, 'celeba')
	if not os.path.exists(dataset_folder + '.zip'):
		print('Downloading zip file from google drive...')
		_download_file_from_google_drive(
			'1ZkjnT495cMBSConUC5MJjeflywyFR4AV', 
			os.path.join(path, dataset_folder + '.zip')
		)
	if not os.path.exists(dataset_folder):
		print('Unzipping celeba dataset folder...')
		with zipfile.ZipFile(dataset_folder + '.zip', 'r') as zip_ref:
			zip_ref.extractall(dataset_folder)
	if not os.path.exists(os.path.join(dataset_folder, 'celeba/img_align_celeba')):
		print('Unzipping images in celeba dataset...')
		with zipfile.ZipFile(os.path.join(dataset_folder, 'celeba/img_align_celeba.zip'), 'r') as zip_ref:
			zip_ref.extractall(os.path.join(dataset_folder, 'celeba/img_align_celeba'))
		print('[DONE] Returning the dataset object')
	celeba = CelebA(path, split=('train' if train else 'test'), transform=T.ToTensor())
	return celeba

# --------------------------------------------------------
# these are from (altered slightly): 
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def _download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : id }, stream = True)
	# token = _get_confirm_token(response)

	params = { 'id' : id, 'confirm' : 1 }
	response = session.get(URL, params = params, stream = True)

	_save_response_content(response, destination)    

def _save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb+") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

# --------------------------------------------------------


# DATA LOADERS FOR CONSTRUCTING BATCHES
# -------------------------------------------------------
# adapted from https://pytorch.org/vision/main/_modules/torchvision/datasets/celeba.html#CelebA
CSV = namedtuple("CSV", ["header", "index", "data"])
class CelebA(VisionDataset):
	def __init__(
		self,
		root: str,
		split: str = "train",
		target_type: Union[List[str], str] = "identity",
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
	) -> None:
		super().__init__(root, transform=transform, target_transform=target_transform)
		self.base_folder = os.path.join(root, 'celeba/celeba')
		self.split = split
		if isinstance(target_type, list):
				self.target_type = target_type
		else:
				self.target_type = [target_type]

		if not self.target_type and self.target_transform is not None:
				raise RuntimeError("target_transform is specified but target_type is empty")

		split_map = {
				"train": 0,
				"valid": 1,
				"test": 2,
				"all": None,
		}
		split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
		splits = self._load_csv("list_eval_partition.txt")
		identity = self._load_csv("identity_CelebA.txt")
		bbox = self._load_csv("list_bbox_celeba.txt", header=1)
		landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
		attr = self._load_csv("list_attr_celeba.txt", header=1)

		mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

		if mask == slice(None):  # if split == "all"
				self.filename = splits.index
		else:
				self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
		self.identity = identity.data[mask]
		self.bbox = bbox.data[mask]
		self.landmarks_align = landmarks_align.data[mask]
		self.attr = attr.data[mask]
		# map from {-1, 1} to {0, 1}
		self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
		self.attr_names = attr.header

	def _load_csv(
			self,
			filename: str,
			header: Optional[int] = None,
	) -> CSV:
			with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
					data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

			if header is not None:
					headers = data[header]
					data = data[header + 1 :]
			else:
					headers = []

			indices = [row[0] for row in data]
			data = [row[1:] for row in data]
			data_int = [list(map(int, i)) for i in data]

			return CSV(headers, indices, torch.tensor(data_int))


	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba/img_align_celeba", self.filename[index]))

		target: Any = []
		for t in self.target_type:
			if t == "attr":
				target.append(self.attr[index, :])
			elif t == "identity":
				target.append(self.identity[index, 0])
			elif t == "bbox":
				target.append(self.bbox[index, :])
			elif t == "landmarks":
				target.append(self.landmarks_align[index, :])
			else:
				# TODO: refactor with utils.verify_str_arg
				raise ValueError(f'Target type "{t}" is not recognized.')

		if self.transform is not None:
			X = self.transform(X)

		if target:
			target = tuple(target) if len(target) > 1 else target[0]

			if self.target_transform is not None:
				target = self.target_transform(target)
		else:
			target = None

		return X, target


	def __len__(self) -> int:
		return len(self.attr)

	def extra_repr(self) -> str:
		lines = ["Target type: {target_type}", "Split: {split}"]
		return "\n".join(lines).format(**self.__dict__)
# -------------------------------------------------------



# -------------------------------------------------------
# adapted from https://pytorch.org/vision/main/_modules/torchvision/datasets/celeba.html#CelebA
CSV = namedtuple("CSV", ["header", "index", "data"])
class CelebA(VisionDataset):
	def __init__(
		self,
		root: str,
		split: str = "train",
		target_type: Union[List[str], str] = "identity",
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
	) -> None:
		super().__init__(root, transform=transform, target_transform=target_transform)
		self.base_folder = os.path.join(root, 'celeba/celeba')
		self.split = split
		if isinstance(target_type, list):
				self.target_type = target_type
		else:
				self.target_type = [target_type]

		if not self.target_type and self.target_transform is not None:
				raise RuntimeError("target_transform is specified but target_type is empty")

		split_map = {
				"train": 0,
				"valid": 1,
				"test": 2,
				"all": None,
		}
		split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
		splits = self._load_csv("list_eval_partition.txt")
		identity = self._load_csv("identity_CelebA.txt")
		bbox = self._load_csv("list_bbox_celeba.txt", header=1)
		landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
		attr = self._load_csv("list_attr_celeba.txt", header=1)

		mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

		if mask == slice(None):  # if split == "all"
				self.filename = splits.index
		else:
				self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
		self.identity = identity.data[mask]
		self.bbox = bbox.data[mask]
		self.landmarks_align = landmarks_align.data[mask]
		self.attr = attr.data[mask]
		# map from {-1, 1} to {0, 1}
		self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
		self.attr_names = attr.header

	def _load_csv(
			self,
			filename: str,
			header: Optional[int] = None,
	) -> CSV:
			with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
					data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

			if header is not None:
					headers = data[header]
					data = data[header + 1 :]
			else:
					headers = []

			indices = [row[0] for row in data]
			data = [row[1:] for row in data]
			data_int = [list(map(int, i)) for i in data]

			return CSV(headers, indices, torch.tensor(data_int))


	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba/img_align_celeba", self.filename[index]))

		target: Any = []
		for t in self.target_type:
			if t == "attr":
				target.append(self.attr[index, :])
			elif t == "identity":
				target.append(self.identity[index, 0])
			elif t == "bbox":
				target.append(self.bbox[index, :])
			elif t == "landmarks":
				target.append(self.landmarks_align[index, :])
			else:
				# TODO: refactor with utils.verify_str_arg
				raise ValueError(f'Target type "{t}" is not recognized.')

		if self.transform is not None:
			X = self.transform(X)

		if target:
			target = tuple(target) if len(target) > 1 else target[0]

			if self.target_transform is not None:
				target = self.target_transform(target)
		else:
			target = None

		return X, target


	def __len__(self) -> int:
		return len(self.attr)

	def extra_repr(self) -> str:
		lines = ["Target type: {target_type}", "Split: {split}"]
		return "\n".join(lines).format(**self.__dict__)
# -------------------------------------------------------


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
		self.batch_size = batch_size
		self.encoder_amount = encoder_amount
		self.device = device
		self.transform = T.Compose(
			[T.ToTensor(), transform] if transform else [T.ToTensor()]
		)
		self.data_set = dset.MNIST(path, train=train)
		self.data_length = len(self.data_set)

	def __iter__(self):
		'''starts MnistLoader iterator'''
		self.count = 0
		return self

	def __len__(self):
		return math.ceil(self.data_length / (2 * self.batch_size + self.encoder_amount))

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
		F_A = torch.stack(F_A).to(self.device).float()
		I_A = torch.stack(I_A).to(self.device).float()
		I_B = torch.stack(I_B).to(self.device).float()
		return (F_A, I_A, I_B)


class CelebALoader:
	def __init__(
		self, 
		encoder_amount=3, 
		batch_size=32, 
		transform=None, 
		path=DATA_PATH, 
		train=True, 
		device=DEVICE,
		scale=True,
	):
		dataset_transform = T.Compose([T.Resize((128, 128)), T.ToTensor()]) if scale else T.ToTensor()
		self.dataset = CelebA(path, 'train' if train else 'test', transform=dataset_transform,)
		self.device = device
		self.transform = transform
		self.batch_size = batch_size
		self.encoder_amount = encoder_amount
		self.classes = {}
		self.identities = []
		for idx in trange(len(self.dataset), desc='saving class indices'):
			_, identity = self.dataset[idx]
			if identity not in self.classes.keys():
				self.classes[identity] = []
				self.identities.append(identity)
			self.classes[identity].append(idx)

	def __len__(self):
		return math.ceil(len(self.dataset) / (2 * self.batch_size + self.encoder_amount))

	def __iter__(self):
		self.count = 0
		return self

	def __next__(self): # TODO: add in transforms
		if self.count >= self.__len__():
			raise StopIteration
		A_idnty_idx = torch.randint(0, len(self.identities), (1,))[0]
		A_idnty = self.identities[A_idnty_idx]
		F_A_idxs = torch.randint(0, len(self.classes[A_idnty]), (self.encoder_amount,))
		F_A = []
		for idx in F_A_idxs:
			x_i = self.dataset[self.classes[A_idnty][idx]][0]
			if self.transform:
				x_i = self.transform(x_i)
			F_A.append(x_i)
		F_A = torch.stack(F_A).to(self.device)
		I_A_idxs = torch.randint(0, len(self.classes[A_idnty]), (self.batch_size,))
		I_A = []
		for idx in I_A_idxs:
			x_i = self.dataset[self.classes[A_idnty][idx]][0]
			if self.transform:
				x_i = self.transform(x_i)
			I_A.append(x_i)
		I_A = torch.stack(I_A).to(self.device)
		I_B = []
		while len(I_B) < self.batch_size:
			random_indices = torch.randint(len(self.dataset), (100,))
			for i in random_indices:
				x_i, y_i = self.dataset[i]
				if y_i != A_idnty:
					if self.transform:
						x_i = self.transform(x_i)
					I_B.append(x_i)
				if len(I_B) == self.batch_size:
					break
		I_B = torch.stack(I_B).to(self.device)
		self.count += 1
		return (F_A, I_A, I_B)


if __name__ == '__main__':

	# test loading in data
	print('  TEST download_mnist_data and download_msceleb')
	mnist_data = download_mnist_data()
	celeba_data = download_celeba()
	assert(len(celeba_data) == 162770)
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
	for i, data in enumerate(dataloader):
		f_A, I_A, I_B = data
		assert(f_A.dtype == torch.float)
		assert(I_A.dtype == torch.float)
		assert(I_B.dtype == torch.float)
		if i >= 5:
			break
	print('    MnistLoader returns batch with correct datatype')
	import time
	start = time.time()
	my_num = 0
	for i, data in enumerate(dataloader):
		my_num += 1 # do a small amount of work
		my_num -= 1
	end = time.time()
	print(f'    One epoch completed in {end - start} seconds')
	assert(end - start < 8)
	print('  ...PASSED')

	# test celebA loader
	print('  TEST CelebALoader')
	n, N = 3, 16
	dataloader = CelebALoader(encoder_amount=n, batch_size=N, scale=False)
	for i, data in enumerate(dataloader):
		f_A, I_A, I_B = data
		assert(f_A.shape == (n, 3, 218, 178))
		assert(I_A.shape == (N, 3, 218, 178))
		assert(I_B.shape == (N, 3, 218, 178))
		if i >= 5:
			break
	print('    CelebALoader returns batch with correct shape')
	for i, data in enumerate(dataloader):
		f_A, I_A, I_B = data
		assert(f_A.dtype == torch.float)
		assert(I_A.dtype == torch.float)
		assert(I_B.dtype == torch.float)
		if i >= 5:
			break
	print('    CelebALoader returns batch with correct datatype')
	import time
	start = time.time()
	my_num = 0
	for i, data in enumerate(dataloader):
		my_num += 1 # do a small amount of work
		my_num -= 1
	end = time.time()
	print(f'    One epoch completed in {end - start} seconds')
	assert(end - start < 110)
	print('  ...PASSED')

