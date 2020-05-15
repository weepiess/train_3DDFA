import os
import shutil
import sys
import net.mobilenet_v1 as mobilenet
import torch
import torch.nn as nn
from model_.vdc_loss import VDCLoss
from model_.wpdc_loss import WPDCLoss
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt

        self.input = None
        self.target = None
        self.model = None
        self.optimizer = None
        self.lr = None
        self.writer = SummaryWriter('runs/train_net')
        # if os.path.exists(self.save_dir):
        #     shutil.rmtree(self.save_dir)
        # os.makedirs(self.save_dir)

    def set_input(self, input,gt):
        x = torch.from_numpy(input).cuda().float()
        y = torch.from_numpy(gt).cuda().float()
        a=Variable(y,requires_grad=True)
        self.target = a.cuda(non_blocking=True)
        self.input = x

    def forward(self,x):
        output = self.model(x)
        return output

    def backward_(self,output):
        loss = self.criterion(output,self.target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def print_log_tensorboard(self,name,val,num):
        self.writer.add_scalar(name,val,num)
        
    def print_log_file(self):
        pass



