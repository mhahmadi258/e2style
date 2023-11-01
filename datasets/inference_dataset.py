from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import torch


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		path = self.paths[index]
		img = Image.open(path)
		img = img.convert('RGB') if self.opts.label_nc == 0 else img.convert('L')
  
		img = np.array(img)
  
		from_imgs = list()
		for i in range(1,8):
			from_imgs.append(Image.fromarray(img[:,i*256:(i+1)*256,:]))
   
		if self.transform:
			from_imgs = [self.transform(from_img) for from_img in from_imgs]
   
		from_imgs = torch.stack(from_imgs)

		return from_imgs
