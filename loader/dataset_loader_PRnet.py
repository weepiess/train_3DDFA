import os.path as osp
from pathlib import Path
import numpy as np
import cv2
import torch
import random
import torch.utils.data as data
from loader.dataset_basic import *

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

class PRNetDataset(DatasetBasic):
    def __init__(self, train_data_file):
        super(PRNetDataset, self).__init__(train_data_file)
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.readTrainData()
        self.index = 0
        self.num_data = len(self.train_data_list)

    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip('\n').split('*')
                self.train_data_list.append(item)
            random.shuffle(self.train_data_list)


    def getBatch(self, batch_list):

        imgs = []
        labels = []
        face_ind = np.loadtxt('./Data/uv-data/face_ind.txt').astype(np.int32)
        triangles = np.loadtxt('./Data/uv-data/triangles.txt').astype(np.int32)
        for item in batch_list:
            img = cv2.imread(item[0])
            label = np.load(item[1])
            img_array = np.array(img, dtype=np.float32)
            img_array = np.reshape(img_array,[3,256,256])
            imgs.append(img_array / 256.0 / 1.1)
            label_array = np.array(label, dtype=np.float32)
            label_array = np.reshape(label_array,[3,256,256])
            labels.append(label_array / 256 / 1.1)

        np_label = np.array(labels)
        np_imgs = np.array(imgs)

        return np_imgs,np_label

    def __call__(self, batch_num):
        if (self.index + batch_num) <= self.num_data:
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            input,gt = self.getBatch(batch_list)
            self.index += batch_num

            return input,gt
        else:
            #pass
            self.index = 0
            random.shuffle(self.train_data_list)
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            input,gt = self.getBatch(batch_list)
            self.index += batch_num

            return input,gt
