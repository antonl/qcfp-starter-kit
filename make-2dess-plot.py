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
from pyqcfp.delayed import SimulationCheckpoint
from pyqcfp.runqcfp import QcfpConfig

from pyqcfp.plotting import plot_result, plot_absorption
params = {
    'savefig.dpi': 300,
    'figure.figsize': (12, 9),
    'font.size': 12,
    'xtick.major.size': 3,
    'xtick.minor.size': 3,
    'xtick.major.width': 1,
    'xtick.minor.width': 1,
    'axes.linewidth': 1
}

params.update({
    'ytick.major.size': params['xtick.major.size'],
    'ytick.minor.size': params['xtick.minor.size'],
    'ytick.major.width': params['xtick.major.width'],
    'ytick.minor.width': params['xtick.minor.width'],
    'axes.labelsize': params['font.size'],
})

rcParams.update(params)

def plot_eigenvecs(energies, evecs2, which, path):
    pts_used = energies.shape[0]
    nvecs = energies.shape[1]
    fig = figure()
    ax = fig.add_subplot(111, polar=True)
    for i in range(nvecs):
        ax.scatter(3*np.pi/2*evecs2[:pts_used//2, i, which],
                   energies[:pts_used//2, i])
    ax.set_rlim(14200, 15500)
    ax.set_theta_offset(-np.pi/4)
    ax.set_rlabel_position(np.degrees(3*np.pi/2))
    ax.set_xticklabels(['{:3.0f}%'.format(x) for x in np.linspace(0, 100, 7)] + [''])
    fig.savefig(path, bbox_inches='tight')

def pool_plot_result(*args, pool=None, **kwargs):
    if pool: return pool.submit(plot_result, *args, **kwargs)

    plot_result(*args, **kwargs)

def pool_plot_absorption(*args, pool=None, **kwargs):
    if pool: return pool.submit(plot_absorption, *args, **kwargs)

    plot_absorption(*args, **kwargs)

def pool_plot_eigenvecs(*args, pool=None, **kwargs):
    if pool: return pool.submit(plot_eigenvecs, *args, **kwargs)

    plot_eigenvecs(*args, **kwargs)

@click.command()
@click.argument('path', type=click.Path(file_okay=False, exists=True))
def make_figures(path):
    # look for pump-probe data file
    path = Path(path)

    try:
        ddfile = h5py.File(str(path/'pump-probe.h5'), 'r')
        absfile = h5py.File(str(path/'absorption.h5'), 'r')
    except FileNotFoundError as e:
        print('Datafiles not found in dir {!s}'.format(path))
        return

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
    energies = np.array(ddfile['00000/meta/one band energies'])
    evecs2 = np.array(ddfile['00000/meta/ge eigenvectors'])**2

    # prepare folder for writing things
    figpath = (path / 'figures')
    figpath.mkdir(exist_ok=True)

    # execute the figure plotting in parallel
    pool = ProcessPoolExecutor(max_workers=20)

    # make diagnostic plots to make sure rotational averaging matches analytic
    s = str(figpath / '2d-reference.png')
    pool_plot_result(w1=w1, w3=w3, signal=ddref, show=False, path=s, pool=pool)

    s = str(figpath / '2d-diff-rotaverage.png')
    pool_plot_result(w1=w1, w3=w3, signal=dd.fieldoff-ddref, show=False,
                     path=s, pool=pool)

    s = str(figpath / '2d-fieldon.png')
    pool_plot_result(w1=w1, w3=w3, signal=dd.fieldon, show=False, path=s,
                     pool=pool)

    s = str(figpath / '2d-fieldoff.png')
    pool_plot_result(w1=w1, w3=w3, signal=dd.fieldoff, show=False, path=s,
                     pool=pool)

    s = str(figpath / '2d-stark.png')
    pool_plot_result(w1=w1, w3=w3, signal=dd.fieldon - dd.fieldoff,
                show=False, path=s, pool=pool)

    s = str(figpath / '2d-projection-w3.png')
    pool_plot_absorption(w3=w3[0], signal=-(dd.fieldon - dd.fieldoff).sum(
        axis=1),
                show=False, path=s, pool=pool)

    # do the same for absorption
    tmp = da.from_array(absfile['00000/data'], chunks=(100, 600))
    pts_used = tmp.shape[0]
    rdataon = tmp[:pts_used//2].mean(axis=0)
    rdataoff = tmp[pts_used//2:].mean(axis=0)
    abs = StarkData(*dask.compute(rdataon, rdataoff))
    w3 = np.array(absfile['w3'])

    # make engine dial plots
    for i in range(energies.shape[1]):
        s = str(figpath / '2d-evec{:d}.png'.format(i))
        pool_plot_eigenvecs(energies=energies, evecs2=evecs2, which=i,
                            path=s, pool=pool)

    # ... same
    absref = np.array(absfile['reference'])

    # Plot absorption
    s = str(figpath / 'abs-reference.png')
    pool_plot_absorption(w3=w3, signal=absref, show=False, path=s, pool=pool)

    s = str(figpath / 'abs-diff-rotaverage.png')
    pool_plot_absorption(w3=w3, signal=abs.fieldoff-absref, show=False,
                         path=s, pool=pool)

    s = str(figpath / 'abs-fieldoff.png')
    pool_plot_absorption(w3=w3, signal=abs.fieldoff, show=False, path=s,
                       pool=pool)


    s = str(figpath / 'abs-fieldon.png')
    pool_plot_absorption(w3=w3, signal=abs.fieldon, show=False, path=s,
                        pool=pool)

    s = str(figpath / 'abs-stark.png')
    pool_plot_absorption(w3=w3, signal=abs.fieldon - abs.fieldoff,
                    show=False, path=s, pool=pool)

if __name__ == '__main__':
    make_figures()
