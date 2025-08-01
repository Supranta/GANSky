import numpy as np
import healpy as hp
import torch
import gansky
import sys

configfile = sys.argv[1]
config = gansky.Config(configfile)

gen_mask  = config.gen_mask
disc_mask = config.disc_mask

real_data = gansky.ArrayDataset(config.real_data_file, config.kappa_std)
fake_data = gansky.ArrayDataset(config.fake_data_file, config.kappa_std)

num_bins = config.n_tomo_bins

trainer = gansky.Trainer(num_bins, gen_mask, disc_mask, real_data, fake_data, num_channels=config.num_channels,\
                        ident_loss_hp=config.ident_loss_hp, scale_loss_hp=config.scale_loss_hp, gan_init=config.gan_init,\
                        save_name=config.gan_path, writer_dir=config.gan_dir)
trainer.train(config.gan_train_steps)
