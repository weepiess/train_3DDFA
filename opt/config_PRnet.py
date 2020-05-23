# -*- coding: utf-8 -*-
from opt.base_config import BaseOptions
import argparse
import torch
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
    
class TrainOptions_PRnet(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser_):
        super(TrainOptions_PRnet, self).initialize(parser_)
        parser_.add_argument('--train_path',default='/media/weepies/Seagate Backup Plus Drive/3DMM/3d-pixel/train_final.txt',type=str)
        parser_.add_argument('--eval_path',default='/media/weepies/Seagate Backup Plus Drive/3DMM/3d-pixel/subt_eva.txt',type=str)

        parser_.add_argument('--epochs', default=100, type=int)
        parser_.add_argument('--start-epoch', default=1, type=int)
        parser_.add_argument('--batch_size_eval',default='64',type=int)
        parser_.add_argument('--mask_path',default='./Data/uv-data/uv_weight_mask_gdh.png',type=str)
        parser_.add_argument('--devices_id', default='0', type=str)
        parser_.add_argument('--base-lr', '--learning-rate', default=0.0001, type=float)
        parser_.add_argument('--batch_size', default=16, type=int)

        self.initialized = True
        return parser_

    def get_config(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        config, _ = parser.parse_known_args()

        # set gpu ids, transfrom string to int number
            # args.devices_id = [int(d) for d in args.devices_id.split(',')]
        str_ids = config.devices_id.split(',')
        config.devices_id = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                config.devices_id.append(id)
        if len(config.devices_id) > 0:
            torch.cuda.set_device(config.devices_id[0])

        return config