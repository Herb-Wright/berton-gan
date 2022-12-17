
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import download_mnist_data
from src.data_utils import download_celeba
from experiments.utils import load_berton_gan
import torch
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
from torchvision.transforms import ToPILImage
from random import randint
import numpy as np

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

to_PIL = ToPILImage()

def make_transfer_image(img_name, berton_gan, dataset, grayscale=True):
	imgs = []
	for i in range(12):
		# pick a content image
		content_img, _ = dataset[randint(0, len(dataset))]
		imgs.append(content_img)
		# pick a style image
		style_img, _ = dataset[randint(0, len(dataset))]
		imgs.append(style_img)
		# generate new image
		new_img = berton_gan.generate_image(style_img.to(DEVICE), content_img.to(DEVICE))
		imgs.append(new_img)
	save_image(img_name, imgs, grayscale)

def make_generated_image(img_name, berton_gan, dataset, grayscale=True):
	imgs = []
	for i in range(36):
		# pick a content image
		content_img, _ = dataset[randint(0, len(dataset))]
		# pick a style image
		style_img, _ = dataset[randint(0, len(dataset))]
		# generate new image
		new_img = berton_gan.generate_image(style_img.to(DEVICE), content_img.to(DEVICE))
		imgs.append(new_img)
	save_image(img_name, imgs, grayscale)

def save_image(image_name, images, grayscale=False):
	sqrtn = math.ceil(math.sqrt(len(images)))
	plt.figure(figsize=(sqrtn, sqrtn))
	gs = gridspec.GridSpec(sqrtn, sqrtn)
	gs.update(wspace=0.05, hspace=0.05)
	for i, img in enumerate(images):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		if grayscale:
			ax.imshow(to_PIL(img), cmap='gray')
		else:
			ax.imshow(to_PIL(img))
	plt.savefig(os.path.join(IMAGES_DIR, image_name), bbox_inches='tight')

def _does_img_exist(img_name):
	return os.path.exists(os.path.join(IMAGES_DIR, img_name))

mnist_data_test = download_mnist_data(train=False)

# FUZZY MNIST GAN
# - one image of result looking fuzzy

img_name = f'mnist_transfer_fuzzy.png'
if not _does_img_exist(img_name):
	berton_gan = load_berton_gan(f'mnist_experiment_herb_7/49')
	make_transfer_image(img_name, berton_gan, mnist_data_test)

img_name = f'mnist_generated_fuzzy.png'
if not _does_img_exist(img_name):
	berton_gan = load_berton_gan(f'mnist_experiment_herb_7/49')
	make_generated_image(img_name, berton_gan, mnist_data_test)

# CRISP MNIST GAN
# - image at epochs 0, 4, 14, 29, 49
# - both generated images and transfer
# - latent space representation

for i in [0, 4, 14, 29, 49]:
	img_name = f'mnist_transfer_crisp_{i+1}.png'
	if not _does_img_exist(img_name):
		berton_gan = load_berton_gan(f'mnist_experiment_herb_8/{i}')
		make_transfer_image(img_name, berton_gan, mnist_data_test)

img_name = f'mnist_generated_crisp_50.png'
if not _does_img_exist(img_name):
	berton_gan = load_berton_gan(f'mnist_experiment_herb_8/49')
	make_generated_image(img_name, berton_gan, mnist_data_test)

img_name = f'mnist_latent_crisp_50.png'
if not _does_img_exist(img_name):
	content_img, _ = mnist_data_test[randint(0, len(mnist_data_test))]
	imgs = []
	berton_gan = load_berton_gan(f'mnist_experiment_herb_8/49')
	for i in range(49):
		# pick a style image
		style_img, _ = mnist_data_test[randint(0, len(mnist_data_test))]
		latent_vect = np.array([(i % 7) - 3, (i // 7) - 3])
		# generate new image
		new_img = berton_gan.generate_image_latent(content_img.to(DEVICE), torch.tensor(latent_vect))
		imgs.append(new_img)
	save_image(img_name, imgs, True)


# CelebA attempt

celeba_dataset = None
if not (_does_img_exist('celeba_transfer.png') and _does_img_exist('celeba_generated.png')):
	celeba_dataset = download_celeba(train=False)

img_name = f'celeba_transfer.png'
if not _does_img_exist(img_name):
	berton_gan = load_berton_gan(f'celeba_attempt')
	make_transfer_image(img_name, berton_gan, celeba_dataset)

img_name = f'celeba_generated.png'
if not _does_img_exist(img_name):
	berton_gan = load_berton_gan(f'celeba_attempt')
	make_generated_image(img_name, berton_gan, celeba_dataset)




# from random import randint

# berton_gan.eval()

# imgs = []
# for i in range(12):
# 	# pick a content image
# 	content_img, _ = testing_data[randint(0, len(testing_data))]
# 	imgs.append(content_img)

# 	# pick a style image
# 	style_img, _ = testing_data[randint(0, len(testing_data))]
# 	imgs.append(style_img)

# 	# generate new image
# 	new_img = berton_gan.generate_image(style_img.to(DEVICE), content_img.to(DEVICE))
# 	imgs.append(new_img)

# print(f'columns 1 and 4 are content; columns 2 and 5 are style; columns 3 and 6 are result')
# show_images(imgs, grayscale=True)


