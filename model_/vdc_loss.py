#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from utils.bfm import MorphabelModel
from utils.transform import P2sRt,similarity_transform
from utils import load
def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x.cuda()
_numpy_to_cuda = lambda x: _tensor_to_cuda(torch.from_numpy(x))
_to_tensor = _numpy_to_cuda  # gpu

class VDCLoss(nn.Module):
    def __init__(self, opt_style='all'):
        super(VDCLoss, self).__init__()

        self.bfm_m = MorphabelModel('./Data/BFM.mat')


    def getvtx(self,param):
        param = param*75660.10889952275-4598.915256886237
        par = param.tolist()
        print('param',param.shape)
        pose_param = []
        shape_param = []
        exp_param = []
        l_v = []
        for i in range(param.shape[0]):
            pose_param = (par[i][0:12])
            pp = np.array(pose_param)
            pp = np.reshape(pp,[3,4])
            shape_param = np.array((par[i][12:211])).astype(np.float32)
            exp_param = np.array((par[i][211:240])).astype(np.float32)
            shape_param = np.reshape(shape_param,[199,1])
            exp_param = np.reshape(exp_param,[29,1])
            s_r, R_r, t_r = P2sRt(pp)

            vtx = self.bfm_m.generate_vertices(shape_param,exp_param)
            # X_homo = np.hstack((vtx, np.ones([vtx.shape[0],1])))
            # transform_vtx = np.dot(P,X_homo.T).T
            transform_vtx = similarity_transform(vtx,s_r,R_r,t_r)
            
            l_v.append(transform_vtx)
            #print('vtx: ',vtx)
        nv = np.array(l_v)
        return nv


    def forward_all(self, input, target):
        input_np = input.cpu().numpy()
        target_np = target.cpu().numpy()



        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        N = input.shape[0]
        offset[:, -1] = offsetg[:, -1]
        gt_vertex = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset

        diff = (gt_vertex - vertex) ** 2
        loss = torch.mean(diff)
        return loss


    def forward(self, input, target):
        return self.forward_all(input, target)



def _parse_param_batch(param):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = param[:, 12:211].view(N, -1, 1)
    alpha_exp = param[:, 211:].view(N, -1, 1)
    return p, offset, alpha_shp, alpha_exp


class VDCLoss(nn.Module):
    def __init__(self, opt_style='all'):
        super(VDCLoss, self).__init__()
        self.model = load.load_BFM('./Data/BFM.mat')
        self.u = _to_tensor(np.array(self.model['shapeMU']))

        self.w_shp = _to_tensor(np.array(self.model['shapePC']))
        self.w_exp = _to_tensor(np.array(self.model['expPC']))

        self.w_shp_length = self.w_shp.shape[0] // 3

        np_mean = np.loadtxt('./Data/gt_mean.txt')
        np_std = np.loadtxt('./Data/gt_std.txt')
        self.param_mean =_to_tensor(np_mean).float()
        self.param_std = _to_tensor(np_std).float()
        self.opt_style = opt_style

    def reconstruct_and_parse(self, input, target):
        # reconstruct
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean

        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def forward_all(self, input, target):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        N = input.shape[0]
        offset[:, -1] = offsetg[:, -1]
        gt_vertex = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset

        diff = (gt_vertex - vertex) ** 2
        loss = torch.mean(diff)
        return loss


    def forward(self, input, target):
        return self.forward_all(input, target)




if __name__ == '__main__':
    pass
