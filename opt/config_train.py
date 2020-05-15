# -*- coding: utf-8 -*-
from opt.base_config import BaseOptions
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
    
class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser_):
        parser_.add_argument('--train_path',default='/media/weepies/Seagate Backup Plus Drive/3DMM/posnet/full_data256_256.txt',type=str)
        parser_.add_argument('--eval_path',default='/media/weepies/Seagate Backup Plus Drive/3DMM/posnet/full_data256_256_eval.txt',type=str)

        parser_.add_argument('--epochs', default=100, type=int)
        parser_.add_argument('--start-epoch', default=1, type=int)
        parser_.add_argument('--batch_size',default='2',type=int)
        parser_.add_argument('--batch_size_eval',default='64',type=int)

        parser_.add_argument('--base-lr', '--learning-rate', default=0.00001, type=float)
        parser_.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser_.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
        parser_.add_argument('--print-freq', '-p', default=20, type=int)

        parser_.add_argument('--devices_id', default='0', type=str)

        parser_.add_argument('--milestones', default='3000,4000', type=str)

        parser_.add_argument('--warmup', default=500, type=int)

        parser_.add_argument('--loss_type', default='wpdc', type=str)
        self.initialized = True
        return parser_