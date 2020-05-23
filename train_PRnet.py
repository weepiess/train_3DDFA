import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

import torch
import torch.optim
from net.resfcn256 import ResFCN256

from loader.dataset_loader_PRnet import PRNetDataset

import argparse
from model_.model_PRnet import PRnet_model
from opt.config_PRnet import TrainOptions_PRnet
import math

if __name__ == '__main__':
    trainConfig = TrainOptions_PRnet()
    opt = trainConfig.get_config() 

    #init data set
    train_data = PRNetDataset(opt.train_path)
    eval_data = PRNetDataset(opt.eval_path)

    iters_total_each_epoch = int(math.ceil(1.0 * train_data.num_data / opt.batch_size))
    iters_total_each_epoch_eval = int(math.ceil(1.0 * eval_data.num_data / opt.batch_size_eval))
    #init model
    model = PRnet_model(opt)
    
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        model.train()

        losses = []
        for iters in range(iters_total_each_epoch):
            input,gt = train_data(opt.batch_size)
            model.set_input(input,gt)
            output = model.forward()
            loss = model.backward_(output)

            losses.append(loss.item())
            print('iters: [%d/%d],Epoch: [%d/%d],Loss: [%f/%f]' %(iters,iters_total_each_epoch,epoch,opt.epochs,loss.item(),np.mean(losses)))

            model.print_log_tensorboard('train_loss',np.mean(losses),epoch * iters_total_each_epoch + iters)

        model.eval()
        eval_loss = []
        for eval_iter in range(iters_total_each_epoch_eval):
            input_eval,gt_eval = eval_data(opt.batch_size)
            loss_eval = model.eval_input(input_eval,gt_eval)

            eval_loss.append(loss_eval.item())
        print('eval loss: ',np.mean(eval_loss))

        model.print_log_tensorboard('eval_loss',np.mean(eval_loss),epoch)
        model.save_checkpoint('/home/weichen/PosNet/result0/'+str(epoch)+'checkpoint.pth.tar')

