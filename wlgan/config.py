import yaml
import numpy as np
import healpy as hp

def get_reordered_mask(mask):
    return hp.reorder(mask, r2n=True)

class Config:
    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        self.gen_mask  = get_reordered_mask(np.load(config_args['gen_mask_file']))
        self.disc_mask = get_reordered_mask(np.load(config_args['disc_mask_file']))
        self.real_data_file = config_args['real_data_file']
        self.fake_data_file = config_args['fake_data_file']
        self.n_tomo_bins    = int(config_args['n_tomo_bins'])
        self.gan_path       = config_args['gan_path']
        self.gan_dir        = config_args['gan_dir']
        self.gan_train_steps = int(config_args['gan_train_steps'])
        self.num_channels    = int(config_args['num_channels'])
