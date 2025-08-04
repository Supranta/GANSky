import sys
import numpy as np
import healpy as hp
import h5py as h5
import torch
import gansky
from gansky.model import Generator, compute_avg_mat
from gansky.summaries import *
from gansky.noisy_mocks import *
import pickle
import os
from tqdm.auto import tqdm, trange

configfile = sys.argv[1]
config = gansky.Config(configfile)

nside = config.nside
lognorm_params_path = config.lognorm_params_path

lmax = 3 * nside

num_bins = 4

mask = np.ones(hp.nside2npix(nside), dtype=bool)

lognorm_params = pickle.load(open(lognorm_params_path, 'rb'))

cl = lognorm_params['cl']
y_cl = lognorm_params['y_cl']
shift = lognorm_params['shift']
mu = lognorm_params['mu']

device = torch.device('cpu')

avg_mat      = compute_avg_mat(nside, mask).to(device)
avg_mat_ring = compute_avg_mat(nside, mask, True).to(device)

ckpt = torch.load(config.gan_path, map_location=device)
gen = Generator(config.n_tomo_bins, avg_mat, num_channels=config.num_channels, num_layers=config.num_layers).to(device)
gen.load_state_dict(ckpt['gen_ema'])

gen_ring = Generator(config.n_tomo_bins, avg_mat_ring, num_channels=config.num_channels, num_layers=config.num_layers).to(device)
gen_ring.load_state_dict(ckpt['gen_ema'])

def kappa_ln2gan(k_ln, std, ring_order=False):
    if ring_order:
        x_ln = torch.tensor(k_ln / std[0])[None,...].to(device, dtype=torch.float32)
        x_gan = gen_ring(x_ln)
        k_gan = (x_gan * std).view(num_bins, -1).cpu().numpy()
    else:
        x_ln = torch.tensor(hp.reorder(k_ln, r2n=True) / std[0])[None,...].to(device, dtype=torch.float32)
        x_gan = gen(x_ln)
        k_gan = hp.reorder((x_gan * std).view(num_bins,-1).cpu().numpy(), n2r=True)
    return k_gan

def generate_lognorm_map(nside):
    alm = hp.synalm(y_cl, lmax=lmax, new=False)
    k = np.exp(mu[:,None] + hp.alm2map(alm, nside, pol=False)) - shift[:,None]
    return k

io_dir = config.gan_io_dir

PIX_AREA = hp.nside2pixarea(nside, degrees=True) * 60. * 60.

kappa_bins = generate_kappa_bins(config.kappa_std)
std = config.kappa_std.astype(np.float32)[None, :, None]

nstart = int(sys.argv[2])
nend   = int(sys.argv[3])

for i in trange(nstart, nend):
    with torch.no_grad():
        k_ln = generate_lognorm_map(nside)
       
        k_gan      = kappa_ln2gan(k_ln, std)
        k_gan_ring = kappa_ln2gan(k_ln, std, True)

        with h5.File(io_dir + '/kappa_samples/mock%d.h5'%(i), 'w') as f:
            f['k_ln']   = k_ln
            f['k_gan']  = k_gan
            f['k_gan_ring']  = k_gan_ring
