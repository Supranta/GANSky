import numpy as np

def _singlebin_kappa_1pt_pdf(kappa_map, kappa_bins):
    kappa_pdf, _ = np.histogram(kappa_map, kappa_bins, density=True)
    return kappa_pdf

def get_1pt_pdf(kappa_maps, kappa_bins, masks):
    nbins = kappa_maps.shape[0]
    kappa_pdf = []
    for mask in masks:
        kappa_pdf_patch = []
        for i in range(nbins):
            kappa_pdf_i = _singlebin_kappa_1pt_pdf(kappa_maps[i][mask], kappa_bins[i])
            kappa_pdf_patch.append(kappa_pdf_i)
        kappa_pdf.append(np.array(kappa_pdf_patch))
    return np.array(kappa_pdf)

def generate_kappa_bins(KAPPA_STD, N_KAPPA_BINS=61):
    kappa_bins = []
    nbins = len(KAPPA_STD)
    for i in range(nbins):
        kappa_bins_i = np.linspace(-4. * KAPPA_STD[i], 7. * KAPPA_STD[i], N_KAPPA_BINS)
        kappa_bins.append(kappa_bins_i)  
    return np.array(kappa_bins)

def save_summary_1pt_pdf(survey_grp, kappa_pdf, kappa_bins):
    summary_grp = survey_grp.create_group('1pt_pdf')
    summary_grp['pdf']        = kappa_pdf
    summary_grp['kappa_bins'] = kappa_bins
    kappa_bincentre = 0.5 * (kappa_bins[:,1:] + kappa_bins[:,:-1])
    summary_grp['kappa_bin_centre'] = kappa_bincentre
