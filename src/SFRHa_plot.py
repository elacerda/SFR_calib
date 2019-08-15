#!/usr/bin/python3
import sys
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
fs = 8
bases = [ 'Padova1994.chab', 'Padova1994.salp', 'Padova2000.chab', 'Padova2000.salp' ]
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


if __name__ == '__main__':
    k__bases_Z = {k: None for k in bases}
    Z__bases = {k: None for k in bases}
    for i, b in enumerate(bases):
        file = '%s/%s_calib.pkl' % (sys.argv[1], b)
        df = pd.read_pickle(file)
        Z = df['Z']
        Z_sun = Z.iloc[-2]
        Z__bases[b] = Z
        k__Z = df['k_Msun_yr']
        k__bases_Z[b] = k__Z
        ###############################
        #### fit logZ x k relation ####
        ###############################
        interval_sel = False
        interval_x = [-0.7, 0.5]
        x = Z/Z_sun
        y = k__Z
        sel = x > -999
        if interval_sel:
            sel = (x > interval_x[0]) & (x < interval_x[1])
        X = x[sel]
        Y = y[sel]
        p1d = np.polyfit(X, Y, 1)
        p2d = np.polyfit(X, Y, 2)
        p3d = np.polyfit(X, Y, 3)
        pexp = np.polyfit(np.log(X), Y, 1)
        fit_p = {'p1d': 'p1d': p1d, 'p2d': p2d, 'p3d': p3d, 'pexp': pexp}
        ###############################
        ###############################
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        N_rows, N_cols = 1, 1
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
        x = Z/Z_sun
        print(b, x, k__Z)
        ax.plot(np.log10(x), k__Z, 'o-k', label='orig')
        ax.plot(np.log10(x), np.polyval(p1d, x), 'b--', label='fit 1d')
        ax.plot(np.log10(x), np.polyval(p2d, x), 'g--', label='fit 2d')
        ax.plot(np.log10(x), np.polyval(p3d, x), 'y--', label='fit 3d')
        ax.plot(np.log10(x), np.polyval(pexp, np.log(x)), 'r--', label='fit exp')
        ax.legend(loc=2, fontsize=fs, ncol=1, borderpad=0, frameon=False)
        tick_params = dict(labelsize=fs, axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(**tick_params)
        ax.set_xlabel(r'$\log$ Z/Z$_\odot$', fontsize=fs)
        ax.set_ylabel(r'k [M$_\odot$ yr$^{-1}$]', fontsize=fs)
        # f.tight_layout()
        f.savefig('logZ_k_%s.%s' % (b.replace('.', '_'), outputImgSuffix), dpi=300)
        plt.close(f)
        ###############################
        f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
        bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
        N_rows, N_cols = 1, 1
        gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
        ax = plt.subplot(gs[0])
        x = Z/Z_sun
        ax.plot(x, k__Z, 'o-k', label='orig')
        ax.plot(x, np.polyval(p1d, x), 'b--', label='fit 1d')
        ax.plot(x, np.polyval(p2d, x), 'g--', label='fit 2d')
        ax.plot(x, np.polyval(p3d, x), 'y--', label='fit 3d')
        ax.plot(x, np.polyval(pexp, np.log(x)), 'r--', label='fit exp')
        ax.legend(loc=2, fontsize=fs, ncol=1, borderpad=0, frameon=False)
        tick_params = dict(labelsize=fs, axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(**tick_params)
        ax.set_xlabel(r'Z/Z$_\odot$', fontsize=fs)
        ax.set_ylabel(r'k [M$_\odot$ yr$^{-1}$]', fontsize=fs)
        # f.tight_layout()
        f.savefig('Z_k_%s.%s' % (b.replace('.', '_'), outputImgSuffix), dpi=300)
        plt.close(f)
        ###############################
    ###############################
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    N_rows, N_cols = 1, 1
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    for b in bases:
        ax.plot(np.log10(Z/Z_sun), k__bases_Z[b], 'o-', label=b)
    tick_params = dict(labelsize=fs, axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(**tick_params)
    ax.set_xlabel(r'$\log$ Z/Z$_\odot$', fontsize=fs)
    ax.set_ylabel(r'k [M$_\odot$ yr$^{-1}$]', fontsize=fs)
    ax.legend(loc=2, fontsize=fs-2, borderpad=0, frameon=False)
    # f.tight_layout()
    f.savefig('logZ_k_bases.%s' % outputImgSuffix, dpi=300)
    plt.close(f)
    ###############################
    f = plot_setup(width=latex_column_width, aspect=1/golden_mean)
    bottom, top, left, right = 0.22, 0.95, 0.15, 0.82
    N_rows, N_cols = 1, 1
    gs = gridspec.GridSpec(N_rows, N_cols, left=left, bottom=bottom, right=right, top=top, wspace=0., hspace=0.)
    ax = plt.subplot(gs[0])
    for b in bases:
        ax.plot(Z/Z_sun, k__bases_Z[b], 'o-', label=b)
    tick_params = dict(labelsize=fs, axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(**tick_params)
    ax.set_xlabel(r'Z/Z$_\odot$', fontsize=fs)
    ax.set_ylabel(r'k [M$_\odot$ yr$^{-1}$]', fontsize=fs)
    ax.legend(loc=2, fontsize=fs-2, borderpad=0, frameon=False)
    # f.tight_layout()
    f.savefig('Z_k_bases.%s' % outputImgSuffix, dpi=300)
    plt.close(f)
    ###############################

# plt.clf()
# ax = plt.gca()
# plotConf__Z = [
#     dict(c='b', lw=1), dict(c='g', lw=1), dict(c='r', lw=1),
#     dict(c='y', lw=1), dict(c='k', lw=2), dict(c='c', lw=1),
# ]
# Z = [0.0004, 0.001, 0.004, 0.008, 0.0190, 0.03]
# for i in range(6):
#     _k = k[i]
#     d = plotConf__Z[i]
#     x = elines.loc[m, 'log_Mass_corr']
#     y = elines.loc[m, 'log_L_Ha_cor'] + np.log10(_k)
#     xm, ym = ma_mask_xyz(x, y)
#     rs = runstats(xm.compressed(), ym.compressed(), smooth=True, sigma=1.2, debug=True, gs_prc=True, frac=0.05, poly1d=True)
#     p = [rs.poly1d_median_slope, rs.poly1d_median_intercept]
#     label = r'Z $=\ %.2f\ Z_\odot$' % (Z[i]/0.019)
#     #ax.plot(rs.xS, rs.yS, label=label,  **d)
#     ax.plot(xm.compressed(), np.polyval(p, xm.compressed()), label=label, **d)
# x = elines.loc[m, 'log_Mass_corr']
# y = elines.loc[m, 'lSFR']
# xm, ym = ma_mask_xyz(x, y)
# rs = runstats(xm.compressed(), ym.compressed(), smooth=True, sigma=1.2, debug=True, gs_prc=True, frac=0.05, poly1d=True)
# p = [rs.poly1d_median_slope, rs.poly1d_median_intercept]
# #ax.plot(rs.xS, rs.yS, c='k', lw=4, ls='--', label='orig')
# ax.plot(xm.compressed(), np.polyval(p, xm.compressed()), c='k', lw=2, ls='--', label='orig')
# ax.legend(loc=2, fontsize=15, ncol=2, borderpad=0, frameon=False)
# plot_text_ax(ax, r'%d SF galaxies' % (m.astype('int').sum()), 0.99, 0.01, 20, 'bottom', 'right', 'k')
# ax.scatter(elines.loc[m, 'log_Mass_corr'], elines.loc[m, 'lSFR'], c='grey', s=10)
# ax.set_xlabel(r'$\log ({\rm M}_\star/{\rm M}_{\odot})$', fontsize=20)
# ax.set_ylabel(r'$\log ({\rm SFR}_{\rm H\alpha}/{\rm M}_{\odot}/{\rm yr})$', fontsize=20)
# ax.set_xlim([8.5, 11.5])
# ax.set_ylim([-2, 2])
