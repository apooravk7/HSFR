from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class vissketchDataset(Dataset):
    def __init__(self, root_sketch, root_vis, transform=None):
        self.root_sketch = root_sketch
        self.root_vis = root_vis
        self.transform = transform

        self.sketch_images = os.listdir(root_sketch)
        self.vis_images = os.listdir(root_vis)
        self.length_dataset = max(len(self.sketch_images), len(self.vis_images)) # 1000, 1500
        self.sketch_len = len(self.sketch_images)
        self.vis_len = len(self.vis_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        sketch_img = self.sketch_images[index % self.sketch_len]
        vis_img = self.vis_images[index % self.vis_len]

        sketch_path = os.path.join(self.root_sketch, sketch_img)
        vis_path = os.path.join(self.root_vis, vis_img)

        sketch_img = np.array(Image.open(sketch_path).convert("RGB"))
        vis_img = np.array(Image.open(vis_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=sketch_img, image0=vis_img)
            sketch_img = augmentations["image0"]
            vis_img = augmentations["image"]

        return sketch_img, vis_img





