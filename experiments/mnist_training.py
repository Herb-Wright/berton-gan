# from torch.utils.data import DataLoader
import torch
import argparse
import time
from tqdm import trange
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import BertonGan, train_all_at_once, MnistLoader, download_mnist_data
from src.training import train_rotate, train_autoencoder
from utils import model_path_exists, load_berton_gan, save_checkpoint

BATCH_SIZE = 32
ENCODER_AMOUNT = 3
EPOCHS = 5
LR = 1e-3
MOMENTUM = 0
DEFAULT_BASE_NAME = 'mnist_experiment_herb_8'

class _Evaluator():
	def __init__(self, verbose=False):
		self.train_dataset = download_mnist_data(train=True)
		self.test_dataset = download_mnist_data(train=False)
		self.verbose = verbose
	
	def evaluate(self, gan:BertonGan, **kwargs):
		gan.eval()
		if self.verbose:
			print('evaluating model...')
		with torch.no_grad():
			correct = 0
			total = 0
			# build latent vectors
			latent_vectors = torch.zeros((10, 2))
			train_length = len(self.train_dataset)
			for i in trange(train_length, desc='building latent vectors for eval', ncols=100):
				x_i, y_i = self.train_dataset[i]
				latent_vectors[y_i] += gan.face_encoder(x_i)
			latent_vectors = (latent_vectors * 10) / train_length
			# evaluate
			for i in trange(len(self.test_dataset), desc='evaluating on test set', ncols=100):
				x_i, y_i = self.test_dataset[i]
				scores = torch.zeros(10)
				for i in range(10):
					scores[i] = gan.shares_style_latent(x_i, latent_vectors[i])
				if torch.argmax(scores) == y_i:
					correct += 1
				total += 1
		gan.train()
		# return
		if self.verbose:
			print(f'got score {correct} / {total}')
		return correct / total


def _load_last_model(base_name, verbose):
	start_epoch = 0
	# decide whether to load a BertonGan or create one
	if(not model_path_exists(base_name + '/0')):
		print('no model exists, creating one')
		berton_gan = BertonGan('mnist')
	else:
		while model_path_exists(base_name + f'/{start_epoch + 1}'):
			start_epoch += 1
		if verbose:
			print(f'loading model from epoch {start_epoch}')
		berton_gan = load_berton_gan(base_name + f'/{start_epoch}')
		berton_gan.train()
		start_epoch += 1
	return berton_gan, start_epoch

def train_mnist_gan(
	base_name:str=DEFAULT_BASE_NAME,
	epochs:int=EPOCHS,
	save:bool=True,
	verbose:bool=True,
	sleep:int=0,
	lr:float=LR,
	evaluate:bool=False
):
	'''
	trains mnist network and saves it (optionally)

	args:
	- `base_name`: the name where the experiment will be stored
	- `epochs`: the number of epochs to run the experiment
	- `save`: whether or not to save the data
	- `verbose`: whether or not to print out extra stuff to console

	this function returns nothing.
	'''
	berton_gan, start_epoch = _load_last_model(base_name, verbose)
	# make our objects and loop for each epoch
	evaluator = _Evaluator(verbose=verbose)
	dataloader = MnistLoader(ENCODER_AMOUNT, BATCH_SIZE)
	if verbose:
		print(f'training for {epochs} more epochs')
	for epoch in range(start_epoch, start_epoch + epochs):
		metadata = train_all_at_once(
			berton_gan, 
			dataloader,
			epochs=1, 
			evaluator=evaluator.evaluate if evaluate else None,
			verbose=verbose,
			epochs_start=epoch,
			optimizer_options={'lr': LR, 'momentum': MOMENTUM}
		)
		# save our epoch data
		if save:
			if verbose:
				print(f'saving checkpoint for epoch {epoch}...')
			save_checkpoint(berton_gan, {
				'epoch': epoch,
				'metadata': metadata[0]
			}, base_name + f'/{epoch}')
		if verbose and sleep > 0:
			print(f'sleeping for {sleep} seconds...')
			time.sleep(sleep)
			

def train_mnist_autoencoder(
	base_name:str=DEFAULT_BASE_NAME,
	epochs:int=EPOCHS,
	save:bool=True,
	verbose:bool=True,
	sleep:int=0,
	lr:float=LR,
):
	'''trains the image encoder and image decoder as an autoencoder'''
	berton_gan, start_epoch = _load_last_model(base_name, verbose)
	# make our objects and loop for each epoch
	evaluator = _Evaluator(verbose=verbose)
	dataloader = MnistLoader(ENCODER_AMOUNT, BATCH_SIZE)
	if verbose:
		print(f'training for {epochs} more epochs')
	for epoch in range(start_epoch, start_epoch + epochs):
		metadata = train_autoencoder(
			berton_gan, 
			dataloader,
			epochs=1, 
			verbose=verbose,
			epochs_start=epoch,
			optimizer_options={'lr': LR, 'momentum': MOMENTUM}
		)
		# save our epoch data
		if save:
			if verbose:
				print(f'saving checkpoint for epoch {epoch}...')
			save_checkpoint(berton_gan, {
				'epoch': epoch,
				'metadata': metadata[0]
			}, base_name + f'/{epoch}')
		if verbose and sleep > 0:
			print(f'sleeping for {sleep} seconds...')
			time.sleep(sleep)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--base_name',
		'-n',
		default=DEFAULT_BASE_NAME,
		help='this is the name of the experiment (the name of the folder stuff will be stored in)',
	)
	parser.add_argument(
		'--verbose',
		'-v',
		action='store_true',
		help='if present, will run program in verbose mode (more stuff printed out)',
	)
	parser.add_argument(
		'--num_epochs',
		'-e',
		default=EPOCHS,
		type=int,
		help='the number of epochs to run',
	)
	parser.add_argument(
		'--no_save',
		action='store_true',
		help='if present, data from experiment will not be saved',
	)
	parser.add_argument(
		'--sleep_each_epoch',
		default=0,
		type=int,
		help='an amount of seconds to sleep after completing each epoch',
	)
	parser.add_argument(
		'--auto_encoder',
		action='store_true',
		help='if true the gan is trained as an autoencoder'
	)
	parser.add_argument(
		'--lr',
		type=int,
		default=LR,
		help='learning rate to run the testing'
	)
	parser.add_argument(
		'--evaluate',
		action='store_true',
	)

	args = parser.parse_args()

	if args.auto_encoder:
		train_mnist_autoencoder(
			args.base_name,
			save=(not args.no_save),
			epochs=args.num_epochs,
			verbose=args.verbose,
			sleep=args.sleep_each_epoch,
			lr=args.lr
		)
	else:
		train_mnist_gan(
			args.base_name,
			epochs=args.num_epochs,
			save=(not args.no_save),
			verbose=args.verbose,
			sleep=args.sleep_each_epoch,
			lr=args.lr,
			evaluate=args.evaluate
		)

