
#%% Import libraries and define functions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from custom_functions import kitaev_plane, energy
from gui_elements import create_vertical_slider, create_checkbox, create_button, sync_slider, update_synced_sliders

#%% KITAEV 2D PP Energy band
#%matplotlib widget

def compute_energy_band_2d(N_x,N_y,mu,t_x,scp_x, pbx = 0, pby = 0, t_y=-1,scp_y=-1 ):
    #P = theoretical_E2(np.pi*Kx,np.pi*Ky,mu,t_x,scp_x, t_y=None, scp_y=None)
    P = energy(kitaev_plane(N_x,N_y,mu,t_x,scp_x, pbx = 1, pby =1, t_y=t_y, scp_y=scp_y),N_x,-1)[:,:,0].real
    P = np.concatenate((P,P), axis=0)
    P = np.concatenate((P,P), axis=1)
    return P

N_x = 20
N_y = 20

init_mu = 0
init_tx = 0
init_scpx = 0 

init_pbx = 1
init_pby = 1

init_ty = -1
init_scpy = -1

smax = 10
smin = -10
sstep = 1

Kx, Ky = np.meshgrid(np.arange(0,4,2/N_x), np.arange(0,4,2/N_y))
# Kx_ct, Ky_ct = np.meshgrid(np.arange(0,4,0.1/N_x), np.arange(0,4,0.1/N_y))


def plot_on_ax(ax,P,init=False):
    if init == False:
        current_zlim = ax.get_zlim()[1]
        ax.clear()
        ax.set_zlim(0,current_zlim)
    surf = ax.plot_surface(Kx,Ky,P,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('$k_x/\pi$')
    ax.set_ylabel('$k_y/\pi$')
    ax.set_title(f"Eigenvalues for Kitaev Plane (N$_x$={N_x},N$_y$={N_y})")
    if init:
        ax.set_zlim(0,2)
        ax.view_init(elev=90, azim=0)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(7,5))
plot_on_ax(ax,compute_energy_band_2d(N_x,N_y,init_mu,init_tx,init_scpx,1,1),init=True)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.15)

mu_slider = create_vertical_slider(fig, [0.05, 0.25, 0.0225, 0.5], '$\\mu$', init_mu, smin, smax, sstep)
tx_slider = create_vertical_slider(fig, [0.1, 0.25, 0.0225, 0.5], '$t_x$', init_tx, smin, smax, sstep)
scpx_slider = create_vertical_slider(fig, [0.15, 0.25, 0.0225, 0.5], '$\\Delta_x$', init_scpx, smin, smax, sstep, color='gray'); scpx_slider.poly.set_alpha(0.1)
ty_slider = create_vertical_slider(fig, [0.2, 0.25, 0.0225, 0.5], '$t_y = \\Delta_y$', init_tx, smin, smax, sstep, color='gray', active=False); ty_slider.poly.set_alpha(0.1)
checkbox_set = create_checkbox(fig, [0.06, 0.02, 0.2, 0.17], ['Lock $\\Delta_x=t_x$', 'Release $t_y$&$\\Delta_y$'], [True, False])
reset_button = create_button(fig, [0.8, 0.025, 0.1, 0.04], 'Reset')
zoom_in_button = create_button(fig, [0.6, 0.025, 0.13, 0.04], 'Zoom In')
zoom_out_button = create_button(fig, [0.45, 0.025, 0.13, 0.04], 'Zoom Out')


# The function to be called anytime a slider's value changes
def update(val):
    plot_on_ax(ax,compute_energy_band_2d(N_x,N_y,mu_slider.val,tx_slider.val,scpx_slider.val,1,1,checkbox_set.get_status()[1]-1,checkbox_set.get_status()[1]-1))
    fig.canvas.draw_idle()

def lock_scpx_handler(label):
    sync_slider(checkbox_set, tx_slider, scpx_slider, ty_slider)

checkbox_set.on_clicked(lock_scpx_handler)

def update_from_tx_handler(val):
    update_synced_sliders(checkbox_set, tx_slider, scpx_slider, ty_slider)

def reset_sliders(event):
    mu_slider.reset()
    tx_slider.reset()
    scpx_slider.reset()

# Function for zooming in
def zoom_in(event):
    #ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
    ax.set_zlim(0, ax.get_zlim()[1] * 0.9)
    plt.draw()

# Function for zooming out
def zoom_out(event):
    #ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
    ax.set_zlim(0, ax.get_zlim()[1] * 1.1)
    plt.draw()

mu_slider.on_changed(update)
tx_slider.on_changed(update)
scpx_slider.on_changed(update)
tx_slider.on_changed(update_from_tx_handler)
reset_button.on_clicked(reset_sliders)
zoom_in_button.on_clicked(zoom_in)
zoom_out_button.on_clicked(zoom_out)

plt.show()