
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

def show_images(images, grayscale=False):
	sqrtn = int(math.ceil(math.sqrt(len(images))))

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
			plt.imshow(img.squeeze(), cmap='gray')
		else:
			plt.imshow(img.squeeze()) 

