#%% Import libraries and define functions
import numpy as np
import matplotlib.pyplot as plt

from custom_functions import kitaev_plane, energy
#%% KITAEV 2D PO Energy wrt mu/t
#%matplotlib widget # uncomment this line if the interactive plot doesn't work as intended in notebooks
N_x = 60
N_y = 5

init_mu = 0
init_tx = 0
init_scpx = 0

init_pbx = 1
init_pby = 0

init_ty = -1
init_scpy = -1

smax = 5
smin = -5
sstep = 0.1

kx = np.arange(0, 2, 2 / N_x)
mulist = np.arange(-5,5,0.5)
K,M = np.meshgrid(kx,mulist)
Egs_kx_mu = np.zeros(K.shape)

def E(N_x, N_y, mu):
    P = np.sort(energy(kitaev_plane(N_x, N_y, mu, 1, 1, pbx=1), N_x, 0), axis=1)[:,N_y]
    return P

for indmu in range(mulist.shape[0]):
        mu = mulist[indmu]
        Egs_kx_mu[indmu,:] = E(N_x, N_y, mu)
M = np.concatenate([M,M],axis=1)
K = np.concatenate([K-2,K],axis=1)
Egs_kx_mu = np.concatenate([Egs_kx_mu, Egs_kx_mu],axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(K, M, Egs_kx_mu, cmap='viridis')
#surf2 = ax.plot_surface(K, M, -Egs_kx_mu, cmap='viridis')
zero_points = np.where(Egs_kx_mu < 1e-13)
ax.scatter(K[zero_points], M[zero_points], -0.1+Egs_kx_mu[zero_points], color='red', alpha=1, label='Zero Energy Points', zorder=10000000)

ax.set_xlabel('$k_x$')
ax.set_ylabel('$\mu/t$')
ax.set_zlabel('Energy')
ax.set_title(f'PO Kitaev energy bands wrt. $\\mu/t$, (P)$N_x={N_x}$, (O)$N_y={N_y}$')

fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)

plt.show()