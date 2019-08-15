#!/usr/bin/python3
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
#
# Calculate the k for SFR(Halpha) = k . L(Halpha)
#
#     Lacerda@Saco - 9/Jul/2014
#
# Update to remove the usage of pystarlight
#
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator  #, MaxNLocator
import matplotlib.gridspec as gridspec


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['legend.numpoints'] = 1
plotConf__Z = [
    dict(c='b', lw=0.5), dict(c='g', lw=0.5), dict(c='r', lw=0.5),
    dict(c='y', lw=0.5), dict(c='k', lw=2.), dict(c='c', lw=0.5),
]
outputImgSuffix = 'pdf'
latex_ppi = 72.0
latex_column_width_pt = 240.0
latex_column_width = latex_column_width_pt/latex_ppi
latex_text_width_pt = 504.0
latex_text_width = latex_text_width_pt/latex_ppi
golden_mean = 0.5 * (1. + 5**0.5)
fs = 10
bases = [ 'Padova1994.chab', 'Padova1994.salp', 'Padova2000.chab', 'Padova2000.salp' ]
baseFile = sys.argv[1]
# '/home/lacerda/LOCAL/data/h5/Base.bc03.h5'
L_sun = 3.826e+33
h = 6.6260755e-27
c = 29979245800.0
yr_sec = 31556900.0
cmInAA = 1e-8          # cm / AA


def plot_setup(width, aspect, fignum=None, dpi=300, cmap=None):
    if cmap is None:
        cmap = 'inferno_r'
    plotpars = {
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'font.size': 8,
        'axes.titlesize': 10,
        'lines.linewidth': 0.5,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.8,
        'font.family': 'Times New Roman',
        'figure.subplot.left': 0.04,
        'figure.subplot.bottom': 0.04,
        'figure.subplot.right': 0.97,
        'figure.subplot.top': 0.95,
        'figure.subplot.wspace': 0.1,
        'figure.subplot.hspace': 0.25,
        'image.cmap': cmap,
    }
    plt.rcParams.update(plotpars)
    figsize = (width, width * aspect)
    return plt.figure(fignum, figsize, dpi=dpi)


def add_subplot_axes(ax, rect, facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize = x_labelsize)
    subax.yaxis.set_tick_params(labelsize = y_labelsize)
    return subax


class tupperware_none(object):
    def __init__(self):
        pass

    def __getattr__(self, attr):
        r = self.__dict__.get(attr, None)
        return r


def SFR_parametrize_trapz(flux, wl, ages, tSF, wl_lum=6562.8):
    '''
    Find the k parameter in the equation SFR = k [M_sun yr^-1] L(Halpha) [(10^8 L_sun)^-1]

    TODO: blablabla

    Nh__Zt is obtained for all t in AGES differently from Nh__Z, which consists in the number
    of H-ionizing photons from MAX_AGE till today (t < MAX_AGE).
    '''
    import scipy.integrate as spi

    mask_age = ages <= tSF

    y = flux * wl * cmInAA * L_sun / (h * c)

    qh__Zt = np.trapz(y=y, x=wl, axis=2) # 1 / Msol
    Nh__Zt = spi.cumtrapz(y=qh__Zt, x=ages, initial=0, axis=1) * yr_sec
    Nh__Z = np.trapz(y=qh__Zt[:, mask_age], x=ages[mask_age], axis=1) * yr_sec

    k_SFR__Z = (1./0.453) * wl_lum * L_sun * yr_sec / (Nh__Z * h * c) # M_sun / yr

    return qh__Zt, Nh__Zt, Nh__Z, k_SFR__Z


def _mask(a, base_mask, fill_value=0.0):
    a__Zt = np.ma.masked_array(a, dtype=a.dtype)
    a__Zt[~base_mask] = np.ma.masked
    a__Zt.fill_value = fill_value
    return a__Zt


def read_hdf5(base_file, base_group):
    b = tupperware_none()
    try:
        with h5py.File(base_file, mode='r') as f:
            base = f[base_group]
            b.ageBase = base['age_base'][()]
            b.metBase = base['Z_base'][()]
            b.l_ssp = base['wl'][()]
            b.nAges = len(b.ageBase)
            b.nMet = len(b.metBase)
            b.baseMask = base['baseMask'][()]

            b.f_ssp = _mask(base['f_ssp'][()], b.baseMask)
            b.Mstars = _mask(base['Mstars'][()], b.baseMask)
            b.YA_V = _mask(base['YA_V'][()], b.baseMask)
            b.sspfile = _mask(base['sspfile'][()], b.baseMask)
            b.aFe = _mask(base['aFe'][()], b.baseMask)
    except:
        raise Exception('Could not open HDF5 base %s[%s].' % (base_file, base_group))
    return b


if __name__ == '__main__':
    k_SFR__bases_Z = {k: None for k in bases}
    Z__bases = {k: None for k in bases}
    f_all = open('LHa_SFR_conversion.csv', 'w')
    print('BaseConf, Z, k_SFR_Msun_yr, k_SFR_ergs_s, fNH_10Myr, age_fNH_95_Myr, age_fNH_98_Myr, age_fNH_99', file=f_all)
    for i, b in enumerate(bases):
        base = read_hdf5(baseFile, b)
        max_yr = base.ageBase[-1]
        max_yr = 10**6.5
        mask = base.l_ssp <= 912         # Angstrom
        f_ssp = base.f_ssp[:,:,mask]
        l = base.l_ssp[mask]
        qh__Zt, Nh__Zt, Nh__Z, k_SFR__Z = SFR_parametrize_trapz(f_ssp, l, base.ageBase, max_yr)
        k_SFR__bases_Z[b] = k_SFR__Z
        Z__bases[b] = base.metBase
        fout = b + '.csv'
        with open(fout, 'w+') as f:
            print('Z, k_SFR_Msun_yr, k_SFR_ergs_s, fNH_10Myr, age_fNH_95_Myr, age_fNH_98_Myr, age_fNH_99', file=f)
            for i, _Z in enumerate(base.metBase):
                age95 = base.ageBase[np.where(Nh__Zt[i] / Nh__Zt[i, -1] <= 0.95)][-1]
                age98 = base.ageBase[np.where(Nh__Zt[i] / Nh__Zt[i, -1] <= 0.98)][-1]
                age99 = base.ageBase[np.where(Nh__Zt[i] / Nh__Zt[i, -1] <= 0.99)][-1]
                print('%s,%.4f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f' % (b, _Z, k_SFR__Z[i], k_SFR__Z[i]/(1e8 * L_sun)/1e-42, Nh__Z[i]/Nh__Zt[i, -1], age95/1e6, age98/1e6, age99/1e9), file=f_all)
                print('%.4f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f' % (_Z, k_SFR__Z[i], k_SFR__Z[i]/(1e8 * L_sun)/1e-42, Nh__Z[i]/Nh__Zt[i, -1], age95/1e6, age98/1e6, age99/1e9), file=f)
        #############
        # to pickle #
        #############
        data = [[base.metBase[i], k_SFR__Z[i], k_SFR__Z[i]/(1e8 * L_sun)/1e-42, Nh__Z[i]] for i in range(len(base.metBase))]
        df = pd.DataFrame(data, columns=['Z', 'k_Msun_yr', 'k_ergs_s', 'Nh'])
        df.set_index('Z', inplace=True, drop=False)
        df.to_pickle('%s_calib.pkl' % b, protocol=2)
        #############
        ################
        # plot Nh / qh #
        ################
        f = plot_setup(width=latex_text_width, aspect=0.8)
        bottom, top, left, right = 0.1, 0.95, 0.08, 0.95
        gs = gridspec.GridSpec(2, 2, left=left, bottom=bottom, right=right, top=top, wspace=0.25, hspace=0.25)
        # f.set_size_inches(10,10)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, :])
        subpos = [0.48, 0.20, 0.35, 0.45]
        subax = add_subplot_axes(ax2, subpos)
        for iZ, _Z in enumerate(base.metBase):
            ax1.plot(np.ma.log10(base.ageBase), Nh__Zt[iZ, :] / 1e60,
                     c=plotConf__Z[iZ]['c'], lw=plotConf__Z[iZ]['lw'], label=r'Z $=\ %.2f Z_\odot$' % (_Z / base.metBase[-2]))
            ax2.plot(np.ma.log10(base.ageBase), Nh__Zt[iZ, :] / Nh__Zt[iZ, -1],
                     c=plotConf__Z[iZ]['c'], lw=plotConf__Z[iZ]['lw'], label=r'Z $=\ %.2f Z_\odot$' % (_Z / base.metBase[-2]))
            subax.plot(np.ma.log10(base.ageBase), Nh__Zt[iZ, :] / Nh__Zt[iZ, -1],
                       c=plotConf__Z[iZ]['c'], lw=plotConf__Z[iZ]['lw'], label=r'Z $=\ %.2f Z_\odot$' % (_Z / base.metBase[-2]))
            ax3.plot(np.ma.log10(base.ageBase), np.ma.log10(qh__Zt[iZ, :]),
                     c=plotConf__Z[iZ]['c'], lw=plotConf__Z[iZ]['lw'], label=r'Z $=\ %.2f Z_\odot$' % (_Z / base.metBase[-2]))
        ax2.axhline(y=0.95, ls='--', c='k')
        ax1.set_ylim([0, 10.2])
        ax2.set_ylim([0, 1.1])
        subax.set_xlim([6.4, 7.2])
        subax.set_ylim([0.80, 1.05])
        subax.xaxis.set_major_locator(MultipleLocator(0.5))
        subax.xaxis.set_minor_locator(MultipleLocator(0.25))
        subax.yaxis.set_major_locator(MultipleLocator(0.1))
        subax.yaxis.set_minor_locator(MultipleLocator(0.05))
        subax.xaxis.grid(which='minor')
        subax.yaxis.grid(which='minor')
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax1.yaxis.set_major_locator(MultipleLocator(2.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax1.xaxis.grid(which='major')
        ax1.yaxis.grid(which='major')
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax2.yaxis.set_major_locator(MultipleLocator(0.25))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax2.xaxis.grid(which='major')
        ax2.yaxis.grid(which='major')
        ax3.xaxis.set_major_locator(MultipleLocator(1))
        ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax3.yaxis.set_major_locator(MultipleLocator(2))
        ax3.yaxis.set_minor_locator(MultipleLocator(1))
        ax3.xaxis.grid(which='minor')
        ax3.yaxis.grid(which='minor')
        tick_params = dict(labelsize=fs, axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax1.tick_params(**tick_params)
        ax2.tick_params(**tick_params)
        ax3.tick_params(**tick_params)
        subax.tick_params(**tick_params)
        ax1.set_xlabel(r'$\log\ t\ [yr]$', fontsize=fs)
        ax1.set_ylabel(r'$\mathcal{N}_H(t)\ [10^{60}\ \gamma\ M_\odot{}^{-1}]$', fontsize=fs)
        ax2.set_xlabel(r'$\log\ t\ [yr]$', fontsize=fs)
        ax2.set_ylabel(r'$\mathcal{N}_H(t)/\mathcal{N}_H$', fontsize=fs)
        ax3.set_xlabel(r'$\log\ t\ [yr]$', fontsize=fs)
        ax3.set_ylabel(r'$\log\ q_H [s^{-1} M_\odot{}^{-1}]$', fontsize=fs)
        ax3.legend(loc=1, fontsize=fs, ncol=2, borderpad=0, frameon=False)
        f.savefig('Nh_logt_metBase_%s.%s' % (b.replace('.', '_'), outputImgSuffix), dpi=300)
        plt.close(f)
        ################
    f_all.close()
