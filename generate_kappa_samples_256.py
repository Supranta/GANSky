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

kappa_bins = generate_kappa_bins(config.kappa_std)
std = config.kappa_std.astype(np.float32)[None, :, None]


nstart = int(sys.argv[2])
nend   = int(sys.argv[3])

ls = np.arange(3*nside)

def get_unbinned_cl(k):
    cl = np.zeros((num_bins,num_bins,3*nside))
    for i in range(num_bins):
        for j in range(i+1):
            if(i!=j):
                cl_ij = hp.anafast(k[i], k[j])
            else:
                cl_ij = hp.anafast(k[i])
            cl[i,j] = cl_ij
            cl[j,i] = cl_ij
    return cl

def get_binned_cl(cl, ell_bins):
    binned_cl = []
    binned_ls = []
    for i in range(len(ell_bins) - 1):
        select_ls = (ls > ell_bins[i]) & (ls < ell_bins[i+1])
        cl_select = cl[:,:,select_ls]
        ls_select = ls[select_ls]
        w_i = 2 * ls_select + 1

        binned_cl_i = np.sum(cl_select * w_i[None,None], axis=-1) / np.sum(w_i)
        binned_cl.append(binned_cl_i)

        binned_ls_i = np.sum(ls_select * w_i) / np.sum(w_i)
        binned_ls.append(binned_ls_i)
    return np.array(binned_cl), np.array(binned_ls)

def get_cl(k, ell_bins):
    print("Computing cl...")
    cl = get_unbinned_cl(k)
    return get_binned_cl(cl, ell_bins)

N_ELL = 21
ell_bins = np.logspace(np.log10(3), np.log10(3*nside), N_ELL+1).astype(int)

for i in trange(nstart, nend):
    with torch.no_grad():
        k_ln = generate_lognorm_map(nside)
       
        x_ln = torch.tensor(hp.reorder(k_ln, r2n=True) / std[0])[None,...].to(device, dtype=torch.float32)
        x_gan = gen(x_ln)
        k_gan = hp.reorder((x_gan * std).view(num_bins,-1).cpu().numpy(), n2r=True)

        cl_ln, ell_bincentre  = get_cl(k_ln, ell_bins)
        cl_gan, ell_bincentre = get_cl(k_gan, ell_bins)

        with h5.File(io_dir + '/kappa_samples/mock%d.h5'%(i), 'w') as f:
            f['k_ln']   = k_ln
            f['k_gan']  = k_gan
            f['cl_ln']  = cl_ln
            f['cl_gan'] = cl_gan
            f['ell_bincentre'] = ell_bincentre
