from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
import torch
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False):
        super(Preprocessor, self).__init__()
        self.dataset = []#dataset
        for inds, item in enumerate(dataset):
            self.dataset.append(item+(inds,))
        self.root = root
        self.transform = transform
        self.mutual = mutual

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        items = self.dataset[index] # fname, pid,pid1,pid2, camid, inds
        fname, camid, inds =items[0],items[-2],items[-1]
        pids = []
        for i, pid in enumerate(items[1:-2]):
            pids.append(pid)

        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return [img, fname]+ pids+[ camid, inds]


    def _get_mutual_item(self, index):
        items = self.dataset[index]  # fname, pid,pid1,pid2, camid, inds
        fname, camid, inds = items[0], items[-2], items[-1]
        pids = []
        for i, pid in enumerate(items[1:-2]):
            pids.append(pid)
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return [img_1,img_2, fname] + pids + [camid, inds]


class UnsupervisedCamStylePreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, num_cam=8, camstyle_dir='', mutual=False):
        super(UnsupervisedCamStylePreprocessor, self).__init__()
        self.dataset = []#dataset
        for inds, item in enumerate(dataset):
            self.dataset.append(item+(inds,))
        self.root = root
        self.transform = transform
        self.mutual = mutual
        self.num_cam = num_cam
        self.camstyle_root = camstyle_dir


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)



    def _get_single_item(self, index):
        items = self.dataset[index] # fname, pid,pid1,pid2, camid, inds
        fname, camid, inds = items[0],items[-2],items[-1]
        sel_cam = torch.randperm(self.num_cam)[0]
        pids = []
        for i, pid in enumerate(items[1:-2]):
            pids.append(pid)

        if sel_cam == camid:
            fpath = osp.join(self.root, fname)
            img = Image.open(fpath).convert('RGB')
        else:
            if 'msmt' in self.root:
                fname = fname[:-4] + '_fake_' + str(sel_cam.numpy() + 1) + '.jpg'
            else:
                fname = fname[:-4] + '_fake_' + str(camid + 1) + 'to' + str(sel_cam.numpy() + 1) + '.jpg'
            fpath = osp.join(self.camstyle_root, fname)
            img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return [img, fname]+ pids+[ camid, inds]




    def _get_mutual_item(self, index):
        items = self.dataset[index]  # fname, pid,pid1,pid2, camid, inds
        fname, camid, inds = items[0], items[-2], items[-1]
        pids = []
        for i, pid in enumerate(items[1:-2]):
            pids.append(pid)

        fname_im = fname.split('/')[-1]

        sel_cam = torch.randperm(self.num_cam)[0]
        if sel_cam == camid:
            try:
                fpath =  fname
            except:
                import ipdb
                ipdb.set_trace()
            img_1 = Image.open(fpath).convert('RGB')
        else:
            if 'msmt' in fname:
                fname_im = fname_im[:-4] + '_fake_' + str(sel_cam.numpy() + 1) + '.jpg'
            else:
                fname_im = fname_im[:-4] + '_fake_' + str(camid + 1) + 'to' + str(sel_cam.numpy() + 1) + '.jpg'
            fpath = osp.join(self.camstyle_root, fname_im)
            img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return [img_1,img_2, fpath] + pids + [camid, inds]
