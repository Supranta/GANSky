import sys
import numpy as np
import healpy as hp
import h5py as h5
import torch
import wlgan as wlg
from wlgan.model import Generator, compute_avg_mat
from wlgan.summaries import *
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

nmocks = 4

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

os.system('rm -rf ' + io_dir + '/samples')
os.system('mkdir -p ' + io_dir + '/samples')


kappa_bins = generate_kappa_bins(config.kappa_std)
std = config.kappa_std.astype(np.float32)[None, :, None]

for i in trange(nmocks):
    with torch.no_grad():
        k_ln = generate_lognorm_map(nside)
        """
        x_ln = torch.tensor(hp.reorder(k_ln, r2n=True) / std[0])[None,...].to(device, dtype=torch.float32)
        x_gan = gen(x_ln)
        k_gan = hp.reorder((x_gan * std).view(num_bins,-1).cpu().numpy(), n2r=True)
        np.save(io_dir + '/gan_mocks/mock_%d.npy'%(i), k_gan)
        np.save(io_dir + '/ln_mocks/mock_%d.npy'%(i), k_ln)
        """
        kappa_pdf = get_1pt_pdf(k_ln, kappa_bins)
        peak_counts = get_tomo_counts(k_ln, kappa_bins, flag='peak')
        void_counts = get_tomo_counts(k_ln, kappa_bins, flag='void')

        with h5.File(io_dir + '/samples/sample_%d.h5'%(i), 'w') as f:
            save_summary_1pt_pdf(f, kappa_pdf, kappa_bins)
            save_summary_peak_void_count(f, peak_counts, kappa_bins, 'peak')
            save_summary_peak_void_count(f, void_counts, kappa_bins, 'void')
