import os.path as osp
from pathlib import Path
import numpy as np
import cv2
import torch
import random
import torch.utils.data as data


def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

class DatasetBasic(data.Dataset):
    def __init__(self, train_data_file):
        super(DatasetBasic, self).__init__()


    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip('\n').split('*')
                self.train_data_list.append(item)
            random.shuffle(self.train_data_list)

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
