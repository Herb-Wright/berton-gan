# from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import trange
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import BertonGan, train_all_at_once, MnistLoader, download_mnist_data
from utils import model_path_exists, load_berton_gan, save_checkpoint

BATCH_SIZE = 16
ENCODER_AMOUNT = 3
EPOCHS = 5
DEFAULT_BASE_NAME = 'mnist_experiment'

class _Evaluator():
	def __init__(self, verbose=False):
		self.train_dataset = download_mnist_data(train=True)
		self.test_dataset = download_mnist_data(train=False)
		self.verbose = verbose
	
	def evaluate(self, gan:BertonGan, **kwargs):
		if self.verbose:
			print('evaluating model...')
		correct = 0
		total = 0
		# build latent vectors
		latent_vectors = torch.zeros((10, 2))
		train_length = len(self.train_dataset)
		for i in trange(train_length, desc='building latent vectors for eval', ncols=90):
			x_i, y_i = self.train_dataset[i]
			latent_vectors[y_i] += gan.face_encoder(x_i)
		latent_vectors = (latent_vectors * 10) / train_length
		# evaluate
		for i in trange(len(self.test_dataset), desc='evaluating on test set', ncols=90):
			x_i, y_i = self.test_dataset[i]
			scores = torch.zeros(10)
			for i in range(10):
				scores[i] = gan.shares_style_latent(x_i, latent_vectors[i])
			if torch.argmax(scores) == y_i:
				correct += 1
			total += 1
		# return
		if self.verbose:
			print(f'got score {correct} / {total}')
		return correct / total

				

def train_mnist_gan(
	base_name=DEFAULT_BASE_NAME,
	epochs=EPOCHS,
	save=True,
	verbose=True
):
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
		start_epoch += 1
	# make our objects and loop for each epoch
	evaluator = _Evaluator(verbose=verbose)
	dataloader = MnistLoader(ENCODER_AMOUNT, BATCH_SIZE)
	if verbose:
		print(f'training for {epochs} more epochs')
	for epoch in range(start_epoch, start_epoch + epochs):
		# train for an epoch
		metadata = train_all_at_once(
			berton_gan, 
			dataloader,
			1, 
			evaluator=evaluator.evaluate,
			verbose=verbose,
			epochs_start=epoch
		)
		# save our epoch data
		if save:
			if verbose:
				print(f'saving checkpoint for epoch {epoch}...')
			save_checkpoint(berton_gan, {
				'epoch': epoch,
				'metadata': metadata[0]
			}, base_name + f'/{epoch}')
			
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--base_name',
		'-n',
		default=DEFAULT_BASE_NAME
	)
	parser.add_argument(
		'--verbose',
		'-v',
		action='store_true'
	)
	parser.add_argument(
		'--num_epochs',
		'-e',
		default=EPOCHS,
		type=int
	)
	parser.add_argument(
		'--no_save',
		action='store_true'
	)

	args = parser.parse_args()

	train_mnist_gan(
		args.base_name,
		epochs=args.num_epochs,
		save=(not args.no_save),
		verbose=args.verbose
	)

