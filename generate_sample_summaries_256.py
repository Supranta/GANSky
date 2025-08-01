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
import pymaster as nmt

configfile = sys.argv[1]
config = wlg.Config(configfile)

nside = 256
lmax = 3 * nside

num_bins = 4

mask = np.ones(hp.nside2npix(nside), dtype=bool)

lognorm_params = pickle.load(open('./data/nside256/lognorm_params_256.pkl', 'rb'))

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
    k = np.exp(mu[:,None] + hp.alm2map(alm, nside, pol=False)) - shift[:,None]
    return k

io_dir = config.gan_io_dir

PIX_AREA = hp.nside2pixarea(nside, degrees=True) * 60. * 60.

"""
sigma_eps         = 0.261
desy3_nbars       = np.array([1.476,1.479,1.484,1.461])
desy3_shape_noise = sigma_eps / np.sqrt(desy3_nbars * PIX_AREA)    

desy6_nbars       = np.array([2.6175 for i in range(4)])
desy6_shape_noise = sigma_eps / np.sqrt(desy6_nbars * PIX_AREA)    

lsst_nbars        = np.array([2.5 for i in range(4)])
lsst_shape_noise  = sigma_eps / np.sqrt(lsst_nbars * PIX_AREA)    
"""

kappa_bins = generate_kappa_bins(config.kappa_std)
std = config.kappa_std.astype(np.float32)[None, :, None]

#std_des = np.array([0.02740464,0.02764861,0.02811132,0.0287259])
#kappa_bins_desy3 = generate_kappa_bins(std_des)
#std_des_y6 = np.array([0.02066338,0.02100683,0.02166899,0.0222049])
#kappa_bins_desy6 = generate_kappa_bins(std_des_y6)
#std_lsst = np.array([0.02113172,0.02146206,0.02214801,0.02273149])
#kappa_bins_lsst = generate_kappa_bins(std_lsst)

nmt_ell_bins, ell_bins, eff_ell_arr = get_nmt_ell_bins(nside)

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
ones_masks = [np.ones(hp.nside2npix(nside)).astype(bool)]

#des_nmt_masks  = get_nmt_masks(des_masks)
#lsst_nmt_masks = get_nmt_masks(lsst_masks)
ones_nmt_masks = get_nmt_masks(ones_masks)

def compute_summaries(k, kappa_bins, masks, nmt_masks):
    kappa_pdf   = get_1pt_pdf(k, kappa_bins, masks)
    peak_counts = get_tomo_counts(k, kappa_bins, masks, flag='peak')
    void_counts = get_tomo_counts(k, kappa_bins, masks, flag='void')
    pseudo_cl   = get_pseudo_cls(k, nmt_ell_bins, nmt_masks)
    return kappa_pdf, peak_counts, void_counts, pseudo_cl
#    return kappa_pdf, peak_counts, void_counts

def save_summaries(f, grp_name, summaries, kappa_bins):
    kappa_pdf, peak_counts, void_counts, pseudo_cl = summaries
#    kappa_pdf, peak_counts, void_counts = summaries
    save_grp = f.create_group(grp_name)
    save_summary_1pt_pdf(save_grp, kappa_pdf, kappa_bins)
    save_summary_peak_void_count(save_grp, peak_counts, kappa_bins, 'peak')
    save_summary_peak_void_count(save_grp, void_counts, kappa_bins, 'void')
    save_summary_pseudocl(save_grp, pseudo_cl, ell_bins, eff_ell_arr)

def compute_and_save_summaries(f, grp_name, k, kappa_bins, masks, nmt_masks):
    summaries = compute_summaries(k, kappa_bins, masks, nmt_masks)
    save_summaries(f, grp_name, summaries, kappa_bins)

for i in trange(nstart, nend):
    with torch.no_grad():
        k_ln = generate_lognorm_map(nside)
       
        x_ln = torch.tensor(hp.reorder(k_ln, r2n=True) / std[0])[None,...].to(device, dtype=torch.float32)
        x_gan = gen(x_ln)
        k_gan = hp.reorder((x_gan * std).view(num_bins,-1).cpu().numpy(), n2r=True)

        """
        k_noisy_desy3    = get_noisy_KS_maps(k_gan, desy3_shape_noise)
        k_noisy_desy3_LN = get_noisy_KS_maps(k_ln,  desy3_shape_noise)

        k_noisy_desy6    = get_noisy_KS_maps(k_gan, desy6_shape_noise)
        k_noisy_desy6_LN = get_noisy_KS_maps(k_ln,  desy6_shape_noise)
        
        k_noisy_lssty1     = get_noisy_KS_maps(k_gan, lsst_shape_noise)
        k_noisy_lssty1_LN  = get_noisy_KS_maps(k_ln,  lsst_shape_noise)
        """
        with h5.File(io_dir + '/samples/noisy_mock%d.h5'%(i), 'w') as f:
            compute_and_save_summaries(f, 'nonoise',    k_gan, kappa_bins, ones_masks, ones_nmt_masks)
            compute_and_save_summaries(f, 'nonoise_LN', k_ln,  kappa_bins, ones_masks, ones_nmt_masks)
            """
            compute_and_save_summaries(f, 'desy3',    k_noisy_desy3,    kappa_bins_desy3, des_masks, des_nmt_masks)
            compute_and_save_summaries(f, 'desy3_LN', k_noisy_desy3_LN, kappa_bins_desy3, des_masks, des_nmt_masks)

            compute_and_save_summaries(f, 'desy6',    k_noisy_desy6,    kappa_bins_desy6, des_masks, des_nmt_masks)
            compute_and_save_summaries(f, 'desy6_LN', k_noisy_desy6_LN, kappa_bins_desy6, des_masks, des_nmt_masks)

            compute_and_save_summaries(f, 'lssty1',    k_noisy_lssty1,    kappa_bins_lsst, lsst_masks, lsst_nmt_masks)
            compute_and_save_summaries(f, 'lssty1_LN', k_noisy_lssty1_LN, kappa_bins_lsst, lsst_masks, lsst_nmt_masks)
            """
