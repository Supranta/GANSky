import sys
import numpy as np
import healpy as hp
import h5py as h5
import torch
import wlgan as wlg
from wlgan.model import Generator, compute_avg_mat
from wlgan.summaries import *
from wlgan.noisy_mocks import *
import pickle
import os
from tqdm.auto import tqdm, trange

configfile = sys.argv[1]
config = wlg.Config(configfile)

nside = 512
gen_nside = 4 * nside
lmax = 2 * gen_nside

num_bins = 4

mask = np.ones(hp.nside2npix(nside), dtype=bool)

lognorm_params = pickle.load(open('./demo/lognorm_params.pkl', 'rb'))

cl = lognorm_params['cl']
y_cl = lognorm_params['y_cl']
shift = lognorm_params['shift']
mu = lognorm_params['mu']

device = torch.device('cpu')

avg_mat = compute_avg_mat(nside, mask).to(device)

ckpt = torch.load(config.gan_path, map_location=device)
gen = Generator(config.n_tomo_bins, avg_mat, num_channels=config.num_channels).to(device)
gen.load_state_dict(ckpt['gen_ema'])

def generate_lognorm_map(nside):
    alm = hp.synalm(y_cl, lmax=lmax, new=False)
    k = np.exp(mu[:,None] + hp.alm2map(alm, gen_nside, pol=False)) - shift[:,None]
    k = hp.ud_grade(k, nside)
    return k

io_dir = config.gan_io_dir

PIX_AREA = hp.nside2pixarea(nside, degrees=True) * 60. * 60.

sigma_eps         = 0.261
desy3_nbars       = np.array([1.476,1.479,1.484,1.461])
desy3_shape_noise = sigma_eps / np.sqrt(desy3_nbars * PIX_AREA)    

desy6_nbars       = np.array([2.6175 for i in range(4)])
desy6_shape_noise = sigma_eps / np.sqrt(desy6_nbars * PIX_AREA)    

lsst_nbars        = np.array([2.5 for i in range(4)])
lsst_shape_noise  = sigma_eps / np.sqrt(lsst_nbars * PIX_AREA)    


kappa_bins = generate_kappa_bins(config.kappa_std)
std = config.kappa_std.astype(np.float32)[None, :, None]

std_des = np.array([0.02740464,0.02764861,0.02811132,0.0287259])
kappa_bins_desy3 = generate_kappa_bins(std_des)
std_des_y6 = np.array([0.02066338,0.02100683,0.02166899,0.0222049])
kappa_bins_desy6 = generate_kappa_bins(std_des_y6)
std_lsst = np.array([0.01391333,0.01441956,0.01534431,0.01618735])
kappa_bins_lsst = generate_kappa_bins(std_lsst)


nstart = int(sys.argv[2])
nend   = int(sys.argv[3])

def get_patch_masks(filename, n_patches=8):
    masks = []
    with h5.File(filename, 'r') as f:
        for i in range(n_patches):
            mask_i = f['patch%d_mask'%(i+1)][:]
            masks.append(mask_i.astype(bool))
    return masks

#des_masks = get_patch_masks('./data/masks/des_patches.h5')
#lsst_masks = get_patch_masks('./data/masks/lsst_patches.h5', 4)
#ones_masks = [np.ones_like(des_masks[0])]

for i in trange(nstart, nend):
    with torch.no_grad():
        k_ln = generate_lognorm_map(nside)
       
        x_ln = torch.tensor(hp.reorder(k_ln, r2n=True) / std[0])[None,...].to(device, dtype=torch.float32)
        x_gan = gen(x_ln)
        k_gan = hp.reorder((x_gan * std).view(num_bins,-1).cpu().numpy(), n2r=True)

        with h5.File(io_dir + '/kappa_samples/mock%d.h5'%(i), 'w') as f:
            f['k_ln']  = k_ln
            f['k_gan'] = k_gan

