import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyqcfp.runqcfp import render_spectral_density

#plt.rcParams.update(
#    {'figure.figsize': (7, 3.67)}
#)

# units in 1/cm
lambda0 = 35
gamma0 = 40
gammaj = 3.

omegah = 1250
gammah = 500
lambdah = 500

omega = np.arange(1e-4, 10000, 0.5)
Ji = 2*lambda0*omega*gamma0/(omega**2 + gamma0**2)

lbd = lambda0

modes = pd.read_csv('oscillator-modes.csv', names=['wj', 'Sj'], delimiter=' ',
                    skiprows=1)
modes = modes.sort_values(by='wj')
print(modes.tail())
Jii = Ji.copy()
Jiii = Ji.copy()
Jnjp = Ji.copy()
for idx, (wj, Sj) in modes.iterrows():
    if wj > 1000:
        continue

    lambdaj = Sj*wj
    lbd += lambdaj
    ogj = omega*gammaj
    Jii += 2*lambdaj*wj**2*(ogj/((wj**2 - omega**2)**2 + ogj**2))

lambda0 = 40
Jnjp = 2*lambda0*omega*gamma0/(omega**2 + gamma0**2)
Jnjp += 2*np.sqrt(2)*lambdah*omegah**2*(gammah*omega/((omegah**2 -
                                                       omega**2)**2 +
                                                      2*gammah**2*omega**2))

print(lbd)
print(lambda0 + lambdah)
for idx, (wj, Sj) in modes.iterrows():
    lambdaj = Sj*wj
    lbd += lambdaj
    ogj = omega*gammaj
    Jiii += 2*lambdaj*wj**2*(ogj/((wj**2 - omega**2)**2 + ogj**2))

fig = plt.figure(dpi=150, figsize=(7, 3.67))

ax = fig.add_subplot(111)

ji_line, = ax.plot(omega, Ji, ls='--')
jii_line, = ax.plot(omega, Jii)
jiii_line, = ax.plot(omega, Jiii, zorder=-1)
jnjp_line, = ax.plot(omega, Jnjp)

print(ji_line.get_color())
print(jii_line.get_color())
print(jiii_line.get_color())
ax.semilogy()
ax.set_ylim(10e-1, 1e5)
ax.set_xlim(0, 2500)
ax.set_xlabel(r'$\omega$ ($\mathrm{cm}^{-1}$)')
ax.set_ylabel(r"C($\omega$) ($\mathrm{cm}^{-1}$)")
plt.show()

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
fig.tight_layout()

from matplotlib.transforms import TransformedBbox, Affine2D
tight_bbox_raw = ax.get_tightbbox(fig.canvas.get_renderer())
tight_bbox_fraw = fig.get_tightbbox(fig.canvas.get_renderer())
tight_bbox = TransformedBbox(tight_bbox_raw, Affine2D().scale(1./fig.dpi)).get_points()
tight_bbox_f = TransformedBbox(tight_bbox_fraw, Affine2D().scale(1./fig.dpi)).get_points()
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#ax.patch.set_facecolor('lightslategray')
fig.savefig('sims-spectral-densities.png', bbox_inches=extent)
#plt.show()
print('aspect: ', 3.67/7)
with open('Ji-spd.txt', 'w') as f:
    f.write(render_spectral_density(omega, Ji))
with open('Jii-spd.txt', 'w') as f:
    f.write(render_spectral_density(omega, Jii))
with open('Jnjp-spd.txt', 'w') as f:
    f.write(render_spectral_density(omega, Jnjp))
