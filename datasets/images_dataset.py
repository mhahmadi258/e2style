from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import torch
import cv2
import random
import skimage
from skimage import img_as_ubyte

def destruct_images(image_list, noise_prob=0.5, resize_prob=0.5):
    destructed_list = []

    for img_path in image_list:
        img = Image.open(img_path)

        # Randomly replace the image with a fully noised version
        if random.random() < noise_prob:
            # Crop the image with the size of 128x128 randomly
            crop_size = (128, 128)
            crop_position = (random.randint(0, img.width - crop_size[0]), random.randint(0, img.height - crop_size[1]))
            cropped_img = img.crop((crop_position[0], crop_position[1], crop_position[0] + crop_size[0], crop_position[1] + crop_size[1]))

            # Resize the cropped version to 256x256
            resized_img = cropped_img.resize((256, 256))
            destructed_list.append(resized_img)

        # Randomly resize the image
        elif random.random() < resize_prob:
            img = img.resize((128, 128))
            img = img.resize((256, 256))
            destructed_list.append(img)

        else:
            destructed_list.append(img)

    return destructed_list


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def preprocessing_for_restoration(self, from_im, to_im):
		from_im = np.array(from_im)
		from_im = cv2.resize(from_im, (256,256))
		to_im = np.array(to_im)
		if np.random.uniform(0, 1) < 0.5:
			from_im = cv2.flip(from_im, 1)
			to_im = cv2.flip(to_im, 1)
		return from_im, to_im

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.opts.dataset_type == 'ffhq_colorization':
			from_im, to_im = self.preprocessing_for_restoration(from_im, to_im)
			from_im=cv2.cvtColor(from_im, cv2.COLOR_BGR2GRAY)
			from_im = np.expand_dims(from_im, axis=2)
			from_im = np.concatenate((from_im, from_im, from_im), axis=-1)
			from_im = Image.fromarray(from_im.astype('uint8')).convert('RGB')
			to_im = Image.fromarray(to_im.astype('uint8')).convert('RGB')        

		elif self.opts.dataset_type == 'ffhq_denoise':
			from_im, to_im = self.preprocessing_for_restoration(from_im, to_im)
			if random.random()>0.5:
				from_im = skimage.util.random_noise(from_im, mode='gaussian', var=0.01)
			else:
				from_im = skimage.util.random_noise(from_im, mode='s&p')
			from_im = img_as_ubyte(from_im)
			from_im = Image.fromarray(from_im.astype('uint8')).convert('RGB')
			to_im = Image.fromarray(to_im.astype('uint8')).convert('RGB')        
		
		elif self.opts.dataset_type == 'ffhq_inpainting':
			from_im, to_im = self.preprocessing_for_restoration(from_im, to_im)
			a = [np.random.choice([35,220],1)[0], 35]
			b = [np.random.choice([35,70],1)[0], 220]
			c = [b[0]+150, 220]
			triangle = np.array([a, b, c])
			from_im = cv2.fillConvexPoly(from_im, triangle, (0, 0, 0))
			from_im = Image.fromarray(from_im.astype('uint8')).convert('RGB')
			to_im = Image.fromarray(to_im.astype('uint8')).convert('RGB')        
		
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im


class MHImagesDataset(Dataset):

	def __init__(self, source_root, train, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		if train:
			self.source_paths =self.source_paths[:int(len(self.source_paths)*0.9)]
		else:
			self.source_paths = self.source_paths[int(len(self.source_paths)*0.9):]
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		path = self.source_paths[index]
		img = Image.open(path)
		img = img.convert('RGB') if self.opts.label_nc == 0 else img.convert('L')
  
		img = np.array(img)
  
		from_imgs = list()
		for i in range(1,8):
			from_imgs.append(Image.fromarray(img[:,i*256:(i+1)*256,:]))
   
		to_img = Image.fromarray(img[:,:256,:])

		if self.target_transform:
			to_img = self.target_transform(to_img)

		from_imgs = destruct_images(from_imgs,0.1, 0.1)
		if self.source_transform:
			from_imgs = [self.source_transform(from_img) for from_img in from_imgs]
   
		from_imgs = torch.stack(from_imgs)

		return from_imgs, to_img
