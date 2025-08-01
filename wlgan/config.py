import yaml
import numpy as np
import healpy as hp

def get_reordered_mask(mask):
    return hp.reorder(mask, r2n=True)

class Config:
    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        self.set_config_map(config_args['map'])
        self.set_config_data(config_args['data'])
        self.set_config_io(config_args['io'])
        self.set_config_hp(config_args['hyperparameters'])

    def set_config_map(self, config_args_map):
        self.n_tomo_bins = int(config_args_map['n_tomo_bins'])
        self.nside       = int(config_args_map['nside'])

    def set_config_data(self, config_args_data):
        self.gen_mask  = get_reordered_mask(np.load(config_args_data['gen_mask_file']))
        self.disc_mask = get_reordered_mask(np.load(config_args_data['disc_mask_file']))
        self.real_data_file = config_args_data['real_data_file']
        self.fake_data_file = config_args_data['fake_data_file']
        self.lognorm_params_path = config_args_data['lognorm_params_path']
        try:
            self.gan_init   = config_args_data['gan_init']
        except:
            self.gan_init   = None
        kappa_std_split_line = config_args_data['kappa_std'].split(',')
        self.kappa_std       = np.array([float(std) for std in kappa_std_split_line])


    def set_config_io(self, config_args_io):
        self.gan_path   = config_args_io['gan_path']
        self.gan_io_dir = config_args_io['gan_io_dir']
        self.gan_dir    = config_args_io['gan_dir']

    def set_config_hp(self, config_args_hp):
        self.gan_train_steps = int(config_args_hp['gan_train_steps'])
        self.num_channels    = int(config_args_hp['num_channels'])
        self.ident_loss_hp   = float(config_args_hp['ident_loss_hp'])
        self.scale_loss_hp   = float(config_args_hp['scale_loss_hp'])
        try:
            self.nl_scale_loss_hp = float(config_args_hp['nl_scale_loss_hp'])
        except:
            self.nl_scale_loss_hp = 0.



