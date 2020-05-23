import sys
from model_.base_model import BaseModel
import net.mobilenet_v1 as mobilenet
import torch
import torch.nn as nn
from model_.vdc_loss import VDCLoss
from model_.wpdc_loss import WPDCLoss
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from net.resfcn256 import ResFCN256
from model_.prnet_loss import WeightMaskLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PRnet_model(BaseModel):
    def __init__(self, opt):
        super(PRnet_model, self).initialize(opt)
        model = ResFCN256()
        torch.cuda.set_device(0)
        self.model = nn.DataParallel(model,device_ids=[0]).cuda()
        self.model.float()

        self.criterion = WeightMaskLoss(opt.mask_path).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.base_lr, betas=(0.5, 0.999))



    def set_input(self, input,gt):
        x = torch.from_numpy(input).cuda().float()
        y = torch.from_numpy(gt).cuda().float()
        a=Variable(y,requires_grad=True)
        self.target = a.cuda(non_blocking=True)
        self.input = x

    def eval_input(self,input_eval,gt_eval):
        x_eval = torch.from_numpy(input_eval).cpu().float()
        y_eval = torch.from_numpy(gt_eval).cpu()
        a_eval=Variable(y_eval,requires_grad=False)
        target_eval = a_eval.cpu().float()
        with torch.no_grad():
            output_eval = self.model(x_eval).cpu()
            loss_eval = self.criterion(output_eval,target_eval)
        return loss_eval

    def forward(self):
        output = self.model(self.input)
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

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


    def load_checkpoint(self,PATH):
        checkpoint = torch.load(PATH,map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        #loss = checkpoint['loss']
        return model


    def save_checkpoint(self, epoch,filename='checkpoint.pth.tar'):
        torch.save(
                {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                },filename)