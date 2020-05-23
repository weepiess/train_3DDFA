# -*- coding: utf-8 -*-
import argparse
import torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser_):
        pass

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

