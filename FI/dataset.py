import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import glob

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
class DZDataset(Dataset):
    def __init__(self, dataset_name, data_root, batch_size=1):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.data_root = data_root

        self.train_cache = {}
        train_ids = os.listdir(os.path.join(self.data_root, "train"))
        
        for id in train_ids:
            self.train_cache[id] = []
            files = sorted(glob.glob(os.path.join(self.data_root, "train", id, '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0].split('_')[-1]))
            for file in files:
                image = cv2.imread(file)
                self.train_cache[id].append(image)
            print('train', id, len(files))

        self.load_data()
    def __len__(self):
        return 1000 * 1
        
    def load_data(self):
        self.meta_data = self.train_cache

    def crop_center(self, hr, ph, pw):
        ih, iw = hr.shape[0:2]
        lr_patch_h, lr_patch_w = ph, pw
        ph = ih // 2 - lr_patch_h // 2
        pw = iw // 2 - lr_patch_w // 2
        return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]

    def __getitem__(self, index): 
        index = index % len(self.meta_data.keys())

        img0s = []
        gts = []
        img1s = []
        timesteps = []

        # UW image
        # In order to reduce the burden of FI model, we first crop the center of the UW image as input
        img0 = self.meta_data[list(self.meta_data.keys())[index]][0]
        shape = img0.shape
        scale = 0.85 / 0.6
        img0_c = cv2.resize(img0, (int(img0.shape[1]*scale), int(img0.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        img0_c = self.crop_center(img0_c, shape[0], shape[1])

        # W Image
        img1_c = self.meta_data[list(self.meta_data.keys())[index]][-1]

        # UW -> W transition image
        ii = random.randint(1,  33 - 2)  
        timestep = ii / 32
        gt = self.meta_data[list(self.meta_data.keys())[index]][ii]
        gt_c = gt

        img0s.append(img0_c)
        img1s.append(img1_c)
        gts.append(gt_c)
        timesteps.append(timestep)

        img0s = np.array(img0s)
        img1s = np.array(img1s)
        gts = np.array(gts)
        
        if self.dataset_name == 'train':
            if random.uniform(0, 1) < 0.5:  
                img0s = img0s[:, :, :, ::-1]
                img1s = img1s[:, :, :, ::-1]
                gts = gts[:, :, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0s = img0s[:, ::-1]
                img1s = img1s[:, ::-1]
                gts = gts[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0s = img0s[:, :, ::-1]
                img1s = img1s[:, :, ::-1]
                gts = gts[:, :, ::-1]
                
        img0s = torch.from_numpy(img0s.copy()).permute(0, 3, 1, 2)
        img1s = torch.from_numpy(img1s.copy()).permute(0, 3, 1, 2)
        gts = torch.from_numpy(gts.copy()).permute(0, 3, 1, 2)
        timesteps = torch.tensor(timesteps).reshape(gts.shape[0], 1, 1, 1)

        return torch.cat((img0s, img1s, gts), 1), timesteps