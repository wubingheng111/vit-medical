import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = self.load_images()

    def load_images(self):
        # Load images from the data directory
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        return image_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = self.load_image(img_name)

        if self.transform:
            image = self.transform(image)

        return image

    def load_image(self, img_name):
        # Load an image and convert it to a tensor
        image = np.load(img_name)  # Placeholder for actual image loading logic
        return torch.tensor(image, dtype=torch.float32)