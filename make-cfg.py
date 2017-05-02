import numpy as np
import pint
from pathlib import PurePosixPath
from pyqcfp.runqcfp import QcfpConfig
from pyqcfp.delayed import SimulationCheckpoint

q = pint.UnitRegistry()
D = 3.34e-30*q('C m')
A3 = 1.113e-40*q('C m^2 V^-1')
to_wn = lambda x: (x/(q.planck_constant*q.speed_of_light)).to('1/cm')

# setup checkpoint
TMPDIR = PurePosixPath('/dev/shm/simulations/17-03-28/')
#TMPDIR = PurePosixPath('C:/aloukian-project/17-03-22')
SCRATCHDIR = PurePosixPath('/home/aloukian/simulations/17-03-28/')
#SCRATCHDIR = TMPDIR
OUTPUTDIR = SCRATCHDIR

metacfg = SimulationCheckpoint()
cfg = QcfpConfig()

# meta info
cfg.simulation_id = 0
cfg.simulation_group = 'Ji-dimer-ct-mu'
cfg.simulation_type = 'pump-probe'
cfg.simulation_code = 'excitons-2d'
#cfg.simulation_type = 'absorption'
#cfg.simulation_code = 'excitons-abs'
cfg.full_output = False
cfg.analytic_orientational_averaging = False
cfg.return_axes = False

metacfg.tmpdir = str(TMPDIR / cfg.simulation_group)
metacfg.scratchdir = str(SCRATCHDIR / cfg.simulation_group)
metacfg.outputdir = str(OUTPUTDIR / cfg.simulation_group)

metacfg.randstate = [0,[0,],0]
metacfg.chunksize = 1
metacfg.mesh_size = 11,13
metacfg.save_plots = True
metacfg.use_rotations = True
metacfg.use_stark = True
metacfg.use_staticdisorder = False
cfg.stark_perturbation = metacfg.use_stark

cfg.output_directory = '.'

# ------------------------------------------------------------------------------
# Configure the system hamiltonian
# energies are in cm^{-1}

# sites and couplings
cfg.nsites = 3

cfg.system_hamiltonian = [[15260., 150., 45.], # Pd1
                          [150., 15190., 45.], # Pd2
                          [45., 45., 15182.], # CT state
                          ]

cfg.spectral_density_couplings = [1., 1., 1.5]

cfg.k_couplings = [[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]]

cfg.system_type = 'paulionic' # two-level systems

# directions of dipoles
cfg.system_dipoles = [
    [-0.7513666, 0.36257354, 0.55135167], # Pd1
    [0.95124446, 0.0857395,  0.29628147], # Pd2
    [0.78834349, 0.28475419, -0.54537106], # CT state direction
]

# direction of static dipoles
cfg.delta_mu = cfg.system_dipoles

# site disorder configuration
fwhm_factor = 2*np.sqrt(2*np.log(2))
metacfg.static_disorder_musigma = [
    [0, float(95./fwhm_factor)],
    [0, float(95./fwhm_factor)],
    [0, float(190./fwhm_factor)],
]

# Transition dipole magnitude in Debye
d_tr = 4. # Debye
d_tr_ct = 0.0 # Debye

# Static dipole magnitude in Debye
d_st = 1.41 # Debye
d_st_ct = 38. # Debye

# Trace of polarizability
tr_alpha = 0. # A3
tr_alpha_ct = 0. # A3

# ------------------------------------------------------------------------------
# configure bath

cfg.bath_temperature = 53.52
cfg.spectral_density_paths = [
    str(PurePosixPath(metacfg.scratchdir) / 'Ji-spd.txt')
]
cfg.zpl_fwhm = 1.0

lamb_cor = 505.332 # reorganization energy correction
cfg.system_hamiltonian = (np.array(cfg.system_hamiltonian) - lamb_cor*np.eye(
    cfg.nsites)).tolist()

# ------------------------------------------------------------------------------
# configure field orientations

cfg.optical_field_vectors = [[1.1, 0., 0.]]*4

Estark = 1. # in MV/cm
cfg.stark_field_vector = (np.array([0., 1., 0.])*Estark).tolist()

# ------------------------------------------------------------------------------
# configure time delay

cfg.t2_delay = 0.

# ------------------------------------------------------------------------------
# configure plotting

cfg.w1_min = cfg.w3_min = 14250.
cfg.w1_max = cfg.w3_max = 16250.
cfg.nfreqs = 600

# ------------------------------------------------------------------------------
# Scale things to correct units, and put in polarizability

tr_alpha = to_wn(tr_alpha*A3*q('MV/cm')**2).magnitude
tr_alpha_ct = to_wn(tr_alpha_ct*A3*q('MV/cm')**2).magnitude

# scale polarizability
monomer_alpha = (tr_alpha*np.eye(3)/3).tolist()
ct_alpha = (tr_alpha_ct*np.eye(3)/3).tolist()

cfg.delta_alpha = [
    monomer_alpha,
    monomer_alpha,
    ct_alpha
]

# scale static dipoles
d_st = to_wn(d_st*D*q('MV/cm')).magnitude
d_st_ct = to_wn(d_st_ct*D*q('MV/cm')).magnitude
scale_static_dipoles = np.diag([d_st, d_st, d_st_ct])
#scale_static_dipoles = np.diag([d_st, d_st])
cfg.delta_mu = (np.dot(scale_static_dipoles, np.array(cfg.delta_mu))).tolist()

# scale transition dipoles
d_tr = to_wn(d_tr*D*q('MV/cm')).magnitude
d_tr_ct = to_wn(d_tr_ct*D*q('MV/cm')).magnitude

scale_tr_dipoles = np.diag([d_tr, d_tr, d_tr_ct]).tolist()
#scale_tr_dipoles = np.diag([d_tr, d_tr]).tolist()
cfg.system_dipoles = (np.dot(scale_tr_dipoles, np.array(
    cfg.system_dipoles))).tolist()

# scale transition and permanent dipole moments
print(np.linalg.norm(np.array(cfg.system_dipoles), axis=1, ord=2))
print(np.linalg.norm(cfg.delta_mu, axis=1, ord=2))
print([np.trace(x) for x in cfg.delta_alpha])

with open('template-cfg.yaml', 'w') as f:
    cfg.to_yaml(stream=f)

with open('metacfg.yaml', 'w') as f:
    metacfg.to_yaml(stream=f)
