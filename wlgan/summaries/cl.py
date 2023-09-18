import numpy as np
import healpy as hp

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

def save_cl(f, cl):
    summary_grp = f.create_group('cl')
    summary_grp['cl'] = cl
    summary_grp['ls'] = np.arange(cl.shape[-1])
