import numpy as np
import healpy as hp
import pymaster as nmt
import time

def get_cl(hp_map):
    npix  = hp_map.shape[-1]
    nside = hp.npix2nside(npix)
    gen_lmax = 3 * nside
    
    nbins = hp_map.shape[0]
    
    cl = np.zeros((nbins,nbins,gen_lmax))
    for i in range(nbins):
        for j in range(i+1):
            if(i==j):
                cl_ij = hp.sphtfunc.anafast(hp_map[i])
            else:
                cl_ij = hp.sphtfunc.anafast(hp_map[i],hp_map[j])
            cl[i,j] = cl_ij
            cl[j,i] = cl_ij
    return cl

def get_nmt_ell_bins(nside, n_ell_bins=17):
    ell_bins     = np.ceil(np.logspace(np.log10(3), np.log10(2 * nside), n_ell_bins)).astype(int)[1:]
    nmt_ell_bins = nmt.NmtBin.from_edges(ell_bins[:-1], ell_bins[1:])
    eff_ell_arr  = nmt_ell_bins.get_effective_ells()
    return nmt_ell_bins, ell_bins, eff_ell_arr

def get_nmt_masks(masks):
    print("Getting NMT masks....")
    nmt_masks = []
    for mask in masks:
        nmt_mask = nmt.mask_apodization(mask, 10., apotype="C1")
        nmt_masks.append(nmt_mask)
    return nmt_masks

def compute_pseudo_cls(kappa, mask, nmt_ell_bins):
    nbins = kappa.shape[0]
    nmt_kappa_fields = [nmt.NmtField(mask, [kappa[i] * mask.astype(float)]) for i in range(nbins)]
    N_ell = nmt_ell_bins.get_n_bands()
    cls = np.zeros((nbins, nbins, N_ell))
    for i in range(nbins):
        for j in range(i+1):
            cl_ij = nmt.compute_full_master(nmt_kappa_fields[i], nmt_kappa_fields[j], nmt_ell_bins)
            cls[i,j] = cl_ij
            cls[j,i] = cl_ij
    return cls

def get_pseudo_cls(kappa_maps, nmt_ell_bins, masks):
    start_time = time.time()
    cls = []
    for mask in masks:
        cl_patch = compute_pseudo_cls(kappa_maps, mask, nmt_ell_bins)
        cls.append(cl_patch)
    end_time = time.time()
    print("Time taken for pseudo-cl calculation: %2.3f s"%(end_time - start_time))
    return np.array(cls)

def save_summary_pseudocl(save_grp, pseudo_cl, ell_bins, eff_ell_arr):
    summary_grp = save_grp.create_group('pseudo_cl')
    summary_grp['cl'] = pseudo_cl
    summary_grp['ell_bins'] = ell_bins
    summary_grp['eff_ell']  = eff_ell_arr

def save_cl(f, cl):
    summary_grp = f.create_group('cl')
    summary_grp['cl'] = cl
    summary_grp['ls'] = np.arange(cl.shape[-1])


