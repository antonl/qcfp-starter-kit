import h5py
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import toolz
from pathlib import Path
import pyqcfp
from pyqcfp.delayed import (SimulationCheckpoint, CfgIdentity,
                            TagId, ApplyLambda, RotateSDesign,
                            AddIIDStaticDisorder, StarkSplitter,
                            compose_transformations)
from pyqcfp.runqcfp import run_simulation
import scipy.stats as stats
import tqdm

if __name__ == '__main__':
    # ------------------------------------------------------------------------------
    # Set some params and load configuration options
    def calculate_delta_cn(spec, ref):
        return np.max(np.abs(spec - ref))

    def get_stats(calculated, reference):
        X = np.abs(calculated - reference)
        mean = X.mean()
        var = X.var()
        return mean, var

    # check that a job-name directory exists in the work dir and look for
    # configuration there

    chkptpath = Path('metacfg.yaml')

    if chkptpath.exists():
        print('Found checkpoint in `{!s}`'.format(chkptpath))
        chkpt = SimulationCheckpoint.from_yaml_file(str(chkptpath))
    else:
        print('A checkpoint file is required at `{!s}`'.format(str(chkptpath)))
        exit(-1)

    cfgpath = chkptpath.parent / 'template-cfg.yaml'

    if cfgpath.exists():
        cfg = pyqcfp.QcfpConfig.from_yaml_file(str(cfgpath))
        print('Found template QcfpConfig file')
    else:
        print('A template QcfpConfig file is required at `{!s}`'.format(str(cfgpath)))
        exit(-1)

    # start from scratch
    #chkpt.randstate = random.getstate()
    chkpt.usecheckpoint = False
    chkpt.nchunks = 0

    # ------------------------------------------------------------------------------
    # start creating pipeline
    template_gen = CfgIdentity(cfg, count=None)
    def set_tmp(cfg):
        cfg.output_directory = chkpt.tmpdir

    tmp_setter = ApplyLambda(set_tmp)

    id_gen = TagId(start_id=0)

    lst = [template_gen, tmp_setter]

    if chkpt.use_staticdisorder:
        disorder_gen = AddIIDStaticDisorder([stats.norm(mu, sigma) for mu,sigma in
                                             chkpt.static_disorder_musigma])

        lst.append(disorder_gen)

    if chkpt.use_rotations:
        if chkpt.use_stark:
            stark_splitter = StarkSplitter()
            lst.append(stark_splitter)

        mesh_gen = RotateSDesign(mesh_order=chkpt.mesh_size[0],
                                 nrotations=chkpt.mesh_size[1])
        lst.append(mesh_gen)

    lst.append(id_gen)

    # determine how many points to use when distributing simulation over cluster

    if chkpt.use_rotations:
        # if using rotations, have to average over the whole mesh
        if chkpt.use_stark:
            # with stark, we get one mesh for field on and one for field off,
            # take them both
            pts_to_use = 2*mesh_gen.npts
        else:
            pts_to_use = mesh_gen.npts

        if chkpt.use_staticdisorder:
            # if using static disorder, calculate that in outer loop
            pts_for_loop = chkpt.chunksize
        else:
            pts_for_loop = 1
    else:
        # not using rotational averaging, so we're going to take chunksize points
        pts_to_use = chkpt.chunksize
        pts_for_loop = 1

    # ------------------------------------------------------------------------------

    # initialize pool of workers for running qcfp
    pool = ProcessPoolExecutor(max_workers=20)

    # combine the simulation pipeline
    make_pipeline = compose_transformations(*lst)
    cfg_stream = make_pipeline()

    # open a file that the main thread will write to
    dstore_name = '{simtype}.h5'.format(simtype=cfg.simulation_type)
    datastore = h5py.File(dstore_name, 'w')

    # compute a single spectrum to make the *.wrk files
    tmp_pipeline = compose_transformations(template_gen, tmp_setter)
    reference_cfg = list(toolz.take(1, tmp_pipeline()))[0]
    reference_cfg.analytic_orientational_averaging = True
    reference_cfg.use_stark = False
    reference_cfg.return_axes = True
    reference_cfg.full_output = False

    future = pool.submit(run_simulation, reference_cfg)
    simout = future.result()

    # Initialize file based on the first run of the simulator
    dsetshape = simout.signal.shape
    dsetdtype = simout.signal.dtype
    refstore = datastore.create_dataset('reference', data=simout.signal)
    datastore.create_dataset('w3', data=simout.w3)
    datastore.create_dataset('w1', data=simout.w1)
    datastore.create_dataset('cfg', data=cfg.to_yaml())
    datastore.create_dataset('chkpt', data=chkpt.to_yaml())

    # save the metadata
    meta = datastore.create_group('meta')

    metadsets = {}
    for k,v in simout.metainfo.items():
        d = np.array(v)
        print(k, d)
        meta.create_dataset(k, shape=d.shape, data=d)
        metadsets[k] = d.shape, d.dtype

    # ------------------------------------------------------------------------------
    print('Ran reference simulation. Starting averaging.')

    for i in tqdm.trange(pts_for_loop, desc='chunks to run', ncols=100):
        grp = datastore.create_group('{:05d}'.format(i))

        # setup dataset to write chunks to
        shape = (pts_to_use,) + dsetshape
        chunks = (100,) + dsetshape
        dset = grp.create_dataset('data', shape, dtype=dsetdtype, chunks=chunks)

        # setup metadata folder
        meta = grp.create_group('meta')
        for k,(s, dtype) in metadsets.items():
            shape = (pts_to_use,) + s
            chunks = (100,) + s
            meta.create_dataset(k, shape=shape, dtype=dtype, chunks=chunks,
                    compression='gzip', compression_opts=4, shuffle=True)

        futures = pool.map(run_simulation, toolz.take(pts_to_use, cfg_stream))

        for simout in tqdm.tqdm(futures, desc='rotational averages',
                                total=pts_to_use, ncols=100):
            idx = simout.id
            dset[idx, :] = simout.signal
            for k in metadsets.keys():
                meta[k][idx, :] = simout.metainfo[k]

    # ------------------------------------------------------------------------------
    # finished with averaging
    datastore.close()

    print('Averaging completed.')
