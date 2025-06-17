"""
Interactive visualization of the energy bands in the 2D Kitaev model with periodic boundary conditions (PBC) along one direction and open boundary conditions (OBC) along the other.

This script computes the energy spectrum of a finite Kitaev plane with PBC in the x-direction and OBC in the y-direction. It provides an interactive GUI with sliders and checkboxes to vary key physical parameters and observe their effects on the energy band structure.

Sliders:
    - mu: Chemical potential. In the Kitaev chain representation, this corresponds to an on-site term that controls the transition between topological and trivial phases. It plays a central role in tuning whether Majorana edge states appear.
    
    - tx: Hopping amplitude along the x-axis. Represents the electron tunneling strength between adjacent lattice sites in the x-direction. In the Majorana basis, this term contributes to inter-cell coupling between Majorana fermions and competes with pairing to define edge behavior.
    
    - scpx: Superconducting pairing strength along the x-axis. In the Majorana representation, it also contributes to the inter-cell coupling. When scpx equals tx, edge Majorana modes become decoupled from the bulk, leading to robust zero-energy edge states.

    - ty = scpy: Hopping and superconducting pairing in the y-direction. While these could be treated independently, here they are locked together for simplicity. These parameters control the bulk coupling in the transverse direction. Individual sliders will be implemented in the future.

Interactive Features:
    - Checkboxes to lock scpx = tx (Majorana edge mode condition) and enable/disable ty = scpy coupling
    - Reset button to restore all sliders to their initial values
    - Zoom In/Out buttons for energy axis rescaling
"""

#%% Import libraries and define functions
import numpy as np
import matplotlib.pyplot as plt

from custom_functions import kitaev_plane, energy
from gui_elements import create_vertical_slider, create_checkbox, create_button, sync_slider, update_synced_sliders
#%% KITAEV 2D PO Energy band
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

kx = np.arange(0,2,2/N_x)


def compute_bandstructure(N_x,N_y,mu,t_x,scp_x, pbx = 0, pby = 0, t_y=-1,scp_y=-1 ):
    P = np.sort( energy(kitaev_plane(N_x,N_y,mu,t_x,scp_x, pbx=pbx, pby=pby, t_y=t_y, scp_y=scp_y),N_x,0).real, axis=1)
    return P

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
lines = ax.plot(kx, compute_bandstructure(N_x,N_y, init_mu, init_tx, init_scpx, True, False, -1, -1 ) ,'.k' ,lw=2)
ax.set_xlabel('$k/\pi$')
ax.set_ylabel('Eigenvalue')
ax.set_title(f'Kitaev Plane\n($N_x$={N_x} with '+chr(79+init_pbx)+f'BC, $N_y$={N_y} with '+chr(79+init_pby)+'BC)')
ax.set_ylim(-0.1, 0.1)
fig.subplots_adjust(left=0.35, bottom=0.25)

mu_slider = create_vertical_slider(fig, [0.05, 0.25, 0.0225, 0.63], label=r"$\mu$", valinit=init_mu, valmin=smin, valmax=smax, valstep=sstep)
tx_slider = create_vertical_slider(fig, [0.1, 0.25, 0.0225, 0.63], label=r"$t_x$", valinit=init_tx, valmin=smin, valmax=smax, valstep=sstep)
scpx_slider = create_vertical_slider(fig, [0.15, 0.25, 0.0225, 0.63], label=r"$\Delta_x$", valinit=init_scpx, valmin=smin, valmax=smax, valstep=sstep, color="gray", active=False)
ty_slider = create_vertical_slider(fig, [0.2, 0.25, 0.0225, 0.63], label=r"$t_y = \Delta_y$", valinit=init_tx, valmin=smin, valmax=smax, valstep=sstep, color="gray", active=False)
checkbox_set = create_checkbox(fig, [0.06, 0.02, 0.2, 0.17], [r"Lock $\Delta_x=t_x$", r"Release $t_y$&$\Delta_y$"], actives=[True, False])
reset_button = create_button(fig, [0.8, 0.025, 0.1, 0.04], 'Reset')
zoom_in_button = create_button(fig, [0.6, 0.025, 0.13, 0.04], 'Zoom In')
zoom_out_button = create_button(fig, [0.45, 0.025, 0.13, 0.04], 'Zoom Out')

def lock_scpx_handler(label):
    sync_slider(checkbox_set, tx_slider, scpx_slider, ty_slider)

checkbox_set.on_clicked(lock_scpx_handler)

def update_from_tx_handler(val):
    update_synced_sliders(checkbox_set, tx_slider, scpx_slider, ty_slider)

def update(val):
    bands = compute_bandstructure(N_x, N_y, mu_slider.val, tx_slider.val, scpx_slider.val, True, False, checkbox_set.get_status()[1]-1,checkbox_set.get_status()[1]-1)
    for i in range(2*N_y):
        lines[i].set_ydata(bands[:,i])
    ax.set_ylim(1.2*np.min(bands).real-0.1,1.2*np.max(bands).real+0.1)
    fig.canvas.draw_idle()

def reset_sliders(event):
    mu_slider.reset()
    tx_slider.reset()
    scpx_slider.reset()

# Function for zooming in
def zoom_in(event):
    #ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
    ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
    plt.draw()

# Function for zooming out
def zoom_out(event):
    ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
    plt.draw()

mu_slider.on_changed(update)
tx_slider.on_changed(update)
scpx_slider.on_changed(update)
tx_slider.on_changed(update_from_tx_handler)
reset_button.on_clicked(reset_sliders)
zoom_in_button.on_clicked(zoom_in)
zoom_out_button.on_clicked(zoom_out)

plt.show()