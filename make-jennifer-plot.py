import dask
import dask.array as da
import h5py
import numpy as np
import click
from pathlib import Path
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

StarkData = namedtuple('StarkData', ['fieldon', 'fieldoff'])

import matplotlib
matplotlib.use('Agg')

from matplotlib.pyplot import *
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import SymmetricalLogLocator
from cycler import cycler

from pyqcfp.delayed import SimulationCheckpoint
from pyqcfp.runqcfp import QcfpConfig

from pyqcfp.plotting import plot_result, plot_absorption

params = {
    'savefig.dpi': 300,
    'figure.figsize': (12,10),
    'font.size': 12,
    #'xtick.major.size': 3,
    #'xtick.minor.size': 3,
    #'xtick.major.width': 1,
    #'xtick.minor.width': 1,
    'axes.linewidth': 1
}

'''
params.update({
    'ytick.major.size': params['xtick.major.size'],
    'ytick.minor.size': params['xtick.minor.size'],
    'ytick.major.width': params['xtick.major.width'],
    'ytick.minor.width': params['xtick.minor.width'],
    'axes.labelsize': params['font.size'],
})
'''

def normalize(data):
    return data / np.abs(data).max()

def plot_evecs(energies, evecs2, which, path):
    rcParams.update(params)
    pts_used = energies.shape[0]
    nvecs = energies.shape[1]
    fig = figure()
    ax = fig.add_subplot(111, polar=True)
    cyc = cycler(color=['b', 'r', 'g'])

    #for which in range(nvecs):
    for i,prop in zip(range(nvecs), cyc):
        ax.scatter(3*np.pi/2*evecs2[:pts_used//2, i, which],
               energies[:pts_used//2, which], alpha=0.8, **prop)
    ax.set_rlim(14200, 15100)
    ax.set_theta_offset(-np.pi/4)
    ax.set_rlabel_position(np.degrees(3*np.pi/2))
    ax.set_xticklabels(['{:3.0f}%'.format(x) for x in np.linspace(0, 100, 7)] + [''])
    fig.savefig(path, bbox_inches='tight')

def plot_2d(w1, w3, signal, path, invert_w1=False, scale=None, axlim=None):
    signal2 = -1 *signal
    rcParams.update(params)

    # plot in 1000 of wn
    w1 = w1.copy()/1e3
    w3 = w3.copy()/1e3

    if invert_w1:
        w1 = -w1

    if scale is None: # calculate scale, return it in meta
        scale = np.max(np.abs(signal2))
        signal2 /= scale

    # fiddle parameters to make the plots look good
    linthresh = 0.01
    linscale = 0.1
    nlevels = 100
    norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=-1, vmax=1)
    logloc = SymmetricalLogLocator(linthresh=linthresh, base=2)

    levels = logloc.tick_values(-1, 1)

    fig = figure()
    ax = fig.add_subplot(111, aspect='equal', adjustable='box-forced')

    qset = ax.contourf(w1, w3, signal2, nlevels, norm=norm, cmap='RdBu_r')
    ax.contour(w1, w3, signal2, levels=levels, colors='k', alpha=0.4)
    cb = fig.colorbar(qset, ax=ax)

    if axlim:
        ypts = xpts = np.array(sorted(axlim))/1e3
        print('Using limits ', xpts, ypts)
        ax.set_xlim(xpts)
        ax.set_ylim(ypts)
    else:
        ypts = xpts = sorted([np.min(w3), np.max(w3)])
        ax.set_xlim(xpts)
        ax.set_ylim(ypts)

    #ax.grid(True)
    ax.add_artist(Line2D(xpts, ypts, linewidth=2, color='k', alpha=0.5,
                              transform=ax.transData))
    mainpath = Path(path)
    fullpath = mainpath.with_suffix('.full.png')
    cbpath = mainpath.with_suffix('.cb.png')
    boundsinfo = mainpath.with_suffix('.info')

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    cb.ax.xaxis.set_visible(False)
    cb.ax.yaxis.set_visible(False)

    dpi_trans = fig.dpi_scale_trans.inverted()

    main_extent = ax.get_window_extent().transformed(dpi_trans)
    cb_extent = cb.ax.get_window_extent().transformed(dpi_trans)

    fig.savefig(str(mainpath), bbox_inches=main_extent)
    fig.savefig(str(cbpath), bbox_inches=cb_extent)
    fig.savefig(str(fullpath), bbox_inches='tight')

    with boundsinfo.open('w') as f:
        print('Main axis bounds:', file=f)
        print(repr(ax.viewLim.get_points()), file=f)
        print(file=f)

        print('Axis limits:', file=f)
        print(repr(ax.get_xlim()), file=f)
        print(repr(ax.get_ylim()), file=f)
        print(file=f)

        print('Signal scale: ', file=f)
        print(scale, file=f)
        print(file=f)

        print('Colorbar bounds:', file=f)
        print([cb.boundaries[x] for x in [0, -1]], file=f)

@click.command()
@click.argument('path', type=click.Path(file_okay=False, exists=True))
@click.option('--limits', default=(None, None), type=(float, float))
def make_figures(path, limits):
    # look for pump-probe data file
    path = Path(path)
    pool = ProcessPoolExecutor(max_workers=6)
    rcParams.update(params)

    try:
        ddfile = h5py.File(str(path/'pump-probe.h5'), 'r')
        absfile = h5py.File(str(path/'absorption.h5'), 'r')
    except FileNotFoundError as e:
        print('Datafiles not found in dir {!s}'.format(path))
        return

    # execute the figure plotting in parallel
    # calculate average pump-probe
    tmp = da.from_array(ddfile['00000/data'], chunks=(100, 600, 600))
    pts_used = tmp.shape[0]
    rdataon = tmp[:pts_used//2].imag.mean(axis=0)
    rdataoff = tmp[pts_used//2:].imag.mean(axis=0)
    dd = StarkData(*dask.compute(rdataon, rdataoff))
    w3, w1 = np.array(ddfile['w3']), np.array(ddfile['w1'])

    # load ref
    ddref = np.array(ddfile['reference']).imag

    # load evecs
    energies = np.array(ddfile['meta/one band energies'])
    nstates = energies.shape[0]
    evecs2 = np.array(ddfile['meta/ge eigenvectors'])**2
    reorgs = np.array(ddfile['meta/reorganization energy matrix'])
    sbcouplingdiag = np.diag(ddfile['meta/sb coupling diagonal'])
    sbcouplingoffdiag = np.diag(ddfile['meta/sb coupling off-diagonal'])
    redfield = np.array(ddfile['meta/redfield relaxation matrix'])
    reorgs = np.diag(reorgs)[1:nstates+1]
    redfield = np.diag(redfield)[1:nstates+1]
    dephasingmat = np.array(ddfile['meta/lifetime dephasing matrix'])[:,
                   ::2] + \
                   1j*np.array(ddfile['meta/lifetime dephasing matrix'])[:, 1::2]
    imagdeph = dephasingmat[1:nstates+1,0].imag

    fixed_energies = energies - reorgs
    fixed_energies2 = energies - reorgs + imagdeph +6.6

    reorgs_trace = np.diagonal(ddfile['00000/meta/reorganization energy matrix'],
                               axis1=1, axis2=2)[:, 1:nstates+1]
    evecs2_trace = np.array(ddfile['00000/meta/ge eigenvectors'])**2


    energies_trace = np.array(ddfile['00000/meta/one band energies'])
    dephasingmat_trace = np.array(ddfile['00000/meta/lifetime dephasing '
                                         'matrix'])[:,:,::2] + \
                   1j*np.array(ddfile['00000/meta/lifetime dephasing '
                                      'matrix'])[:,:,1::2]
    imagdeph_trace = dephasingmat_trace[:, 1:nstates+1,0].imag
    corr_energies = energies_trace - reorgs_trace + imagdeph_trace + 3.2

    # prepare folder for writing things
    figpath = (path / 'jennifer-figs')
    figpath.mkdir(exist_ok=True)

    with (figpath / 'eigen-energies.info').open('w') as f:
        print('Eigen-energies:', energies, file=f)
        print('GE reorganization energies:', reorgs, file=f)
        print('Shifted energies:', fixed_energies, file=f)
        print('Shifted energies + deph:', fixed_energies2, file=f)
        print(file=f)
        for i in range(evecs2.shape[0]):
            print('Localization of eigenstate {:d}:'.format(i), file=f)
            print(evecs2[i, :], file=f)
            print(file=f)
        print('S-B diagonal couplings:', sbcouplingdiag, file=f)
        print('S-B off-diagonal couplings:', sbcouplingoffdiag, file=f)
        print(file=f)


    # make diagnostic plots to make sure rotational averaging matches analytic
    s = str(figpath / '2d-reference.png')
    pool.submit(plot_2d, w1=w1, w3=w3, signal=ddref, path=s, axlim=limits)

    s = str(figpath / '2d-reference-old.png')
    pool.submit(plot_result, w1=w1, w3=w3, signal=ddref, path=s,
                show=False)

    s = str(figpath / '2d-fieldon.png')
    pool.submit(plot_2d, w1=w1, w3=w3, signal=dd.fieldon, path=s, axlim=limits)

    s = str(figpath / '2d-fieldoff.png')
    pool.submit(plot_2d, w1=w1, w3=w3, signal=dd.fieldoff, path=s, axlim=limits)

    s = str(figpath / '2d-stark.png')
    pool.submit(plot_2d, w1=w1, w3=w3, signal=dd.fieldon-dd.fieldoff, path=s, axlim=limits)

    for i in range(nstates):
        s = str(figpath / '2d-evecs{:d}.png'.format(i))
        pool.submit(plot_evecs, corr_energies, evecs2_trace, i, s)

    dd_projection = -(ddref).sum(axis=1)
    ddess_projection = -(dd.fieldon - dd.fieldoff).sum(axis=1)

    # do the same for absorption
    tmp = da.from_array(absfile['00000/data'], chunks=(100, 600))
    pts_used = tmp.shape[0]
    rdataon = tmp[:pts_used//2].mean(axis=0)
    rdataoff = tmp[pts_used//2:].mean(axis=0)
    abs = StarkData(*dask.compute(rdataon, rdataoff))
    w3 = np.array(absfile['w3'])

    # ... same
    absref = np.array(absfile['reference'])

    # Plot absorption
    s = str(figpath / 'abs-reference.png')

    # make absorption plot
    fig = figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(w3, normalize(absref), linewidth=2)
    ax1.plot(w3, normalize(dd_projection), linewidth=2)
    ax1.set_xlim(*limits)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    #ax1.grid(True)

    dpi_trans = fig.dpi_scale_trans.inverted()
    main_extent = ax1.get_window_extent().transformed(dpi_trans)
    print(main_extent)
    fig.savefig(s, bbox_inches=main_extent)

    boundsinfo = Path(s).with_suffix('.info')
    with boundsinfo.open('w') as f:
        print('Main axis bounds:', file=f)
        print(repr(ax1.viewLim.get_points()), file=f)

    # Plot Stark
    s = str(figpath / 'abs-stark.png')
    fig = figure()
    ax2 = fig.add_subplot(111)

    ax2.plot(w3, normalize(abs.fieldon - abs.fieldoff), linewidth=2)
    ax2.plot(w3, normalize(ddess_projection), linewidth=2)
    ax2.set_xlim(*limits)
    print('Using limits ', limits)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    #ax2.grid(True)

    dpi_trans = fig.dpi_scale_trans.inverted()
    main_extent = ax2.get_window_extent().transformed(dpi_trans)
    print(main_extent)
    fig.savefig(s, bbox_inches=main_extent)

    boundsinfo = Path(s).with_suffix('.info')
    with boundsinfo.open('w') as f:
        print('Main axis bounds:', file=f)
        print(repr(ax2.viewLim.get_points()), file=f)

if __name__ == '__main__':
    make_figures()
