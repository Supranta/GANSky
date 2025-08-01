import numpy as np
import healpy as hp

def get_neighbor_maps(hp_map):
    npix = hp_map.shape[0]
    nside = hp.npix2nside(npix)
    neighbour_indices = hp.get_all_neighbours(nside, np.arange(npix))
    neighbor_maps = []
    for i in range(8):
        neighbor_maps.append(hp_map[neighbour_indices[i]])
    return np.array(neighbor_maps)

def get_kappa_peaks(hp_map, mask):
    neighbor_maps     = get_neighbor_maps(hp_map)
    max_neighbour_map = np.max(neighbor_maps, axis=0)
    select_peaks      = (hp_map > max_neighbour_map) & mask
    return hp_map[select_peaks]

def get_kappa_troughs(hp_map, mask):
    neighbor_maps     = get_neighbor_maps(hp_map)
    min_neighbour_map = np.min(neighbor_maps, axis=0)
    select_troughs      = (hp_map < min_neighbour_map) & mask
    return hp_map[select_troughs]

def get_counts(kappa_map, kappa_bins, mask, flag='peak'):
    if flag=='peak':
        kappa_features = get_kappa_peaks(kappa_map, mask)
    elif flag=='void':
        kappa_features = get_kappa_troughs(kappa_map, mask)
    counts, _ = np.histogram(kappa_features, kappa_bins, density=True)
    return counts

def get_tomo_counts(kappa_maps, kappa_bins, masks, flag='peak'):
    counts_list = []
    nbins = kappa_maps.shape[0]
    for mask in masks:
        counts_list_patch = []
        for i in range(nbins):
            counts = get_counts(kappa_maps[i], kappa_bins[i], mask, flag)
            counts_list_patch.append(counts)
        counts_list.append(np.array(counts_list_patch))
    return np.array(counts_list)

def save_summary_peak_void_count(survey_grp, counts, kappa_bins, flag):
    summary_grp = survey_grp.create_group(flag)
    summary_grp['counts'] = counts
    summary_grp['kappa_bins'] = kappa_bins
    kappa_bincentre = 0.5 * (kappa_bins[:,1:] + kappa_bins[:,:-1])
    summary_grp['kappa_bin_centre'] = kappa_bincentre
