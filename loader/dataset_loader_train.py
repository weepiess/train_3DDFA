import os.path as osp
from pathlib import Path
import numpy as np
import cv2
import torch
import random
import torch.utils.data as data


def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

class DDFADataset(data.Dataset):
    def __init__(self, train_data_file):
        super(DDFADataset, self).__init__()
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.readTrainData()
        self.index = 0
        self.num_data = len(self.train_data_list)
        mean_f = open('./Data/gt_mean.txt')
        std_f = open('./Data/gt_std.txt')
        mean = []
        for line in mean_f.readlines():
            lines = line.strip('\n')
            word = lines.split(' ')
            mean.append(float(word[0]))
        np_mean = np.array(mean)
        mean_f.close()

        std = []
        for line in std_f.readlines():
            lines = line.strip('\n')
            word = lines.split(' ')
            std.append(float(word[0]))
        np_std = np.array(std)
        std_f.close()

        self.mean_val = np_mean
        self.std_val = np_std


    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip('\n').split('*')
                self.train_data_list.append(item)
            random.shuffle(self.train_data_list)

    def normalization(self,param):
        param = (param-self.mean_val)/self.std_val
        return param

    def getBatch(self, batch_list):
        batch = []
        imgs = []
        labels = []
        for item in batch_list:
            img = cv2.imread(item[0])
            ll = []
            f_label = open(item[1])
            for line in f_label.readlines():
                lines = line.strip('\n')
                word = lines.split(' ')
                ll.append(float(word[0]))
            f_label.close()
            #label = (item[1])
            img_array = np.array(img, dtype=np.float32).T
            imgs.append(img_array / 255.0)
            ll_array = np.array(ll, dtype=np.float32)
            labels.append(ll_array)
            # label_array = np.array(label, dtype=np.float32)
            # labels.append(label_array / 256 / 1.1)
        np_label = np.array(labels)
        norm_label = self.normalization(np_label)
        np_imgs = np.array(imgs)
        return np_imgs,norm_label

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
