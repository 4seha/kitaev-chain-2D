
#%% Import libraries and define functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from seaborn import heatmap

from custom_functions import kitaev_plane, cached_eigenstates, ground_eigensolutions
from gui_elements import create_vertical_slider, create_horizontal_slider, create_checkbox, create_camera_sync_checkbox, sync_slider, update_synced_sliders

import time
from functools import partial


#%% KITAEV 2D Eigenvectors
from matplotlib import cm
#%matplotlib widget
from matplotlib.widgets import Slider, CheckButtons

Nx_ticks = (2,10,20)    #Ny_ticks = (2,10,20)

init_mu = 0
init_tx = 0
init_scpx = 0

init_pbx = 1
init_pby = 0

init_ty = -1
init_scpy = -1

smax = 5
smin = -5
sstep = 1

surfaces = [None, None]

def plot_on_axes(surfaces, axes, params, recreate = True):
    N_x, N_y = params[0:2]
    start = time.time()
    # gvals,gvecs = eigenstates(H, cell_count_in_x)
    gvals,gvecs = cached_eigenstates(params)
    print("Eigen calc took", time.time() - start, "seconds")
    gmodes = gvals.shape[0]
    genergy = gvals[0]
    P = np.sum(abs(gvecs)**2,axis=1)
    z_data = []
    for i in range(2):
        z_data.append( P[i::2].reshape((N_x, N_y),order='F') )
    x_mesh,y_mesh = np.meshgrid(range(N_x),range(N_y))
    # ax.plot_surface(y,x,a.T)
    if recreate:    # If data size (Nx or Ny) changes, need to clear and redefine the axes
        if surfaces == [None, None]:    # If initializing
            axes[0].view_init(elev=75, azim=-115)
            axes[1].view_init(elev=75, azim=-115)
        else:   #If not initializing, need to clear the previous axes
            axes[0].clear()
            axes[1].clear()
        for i in range(2):
            surfaces[i] = axes[i].plot_surface(x_mesh, y_mesh, z_data[i].T, color='black', cmap=cm.coolwarm,
                        linewidth=0.5, antialiased=False)
        for ax in axes:
            ax.set_xlabel('$n_x$')
            ax.set_ylabel('$n_y$')
        fig.suptitle(f"Eigenvectors for Kitaev Plane (N$_x$={N_x},N$_y$={N_y})\n{gmodes} modes with energy {genergy.real:.3e}")
        axes[0].set_title("$|\gamma_1|^2$")
        axes[1].set_title("$|\gamma_2|^2$")
    else:   # If data size stays the same, then only changing the surface object is sufficient and more optimal 
        for i in range(2):
            surfaces[i].remove()
            surfaces[i] = axes[i].plot_surface(x_mesh, y_mesh, z_data[i].T, cmap=cm.coolwarm,
                                                    linewidth=0.5, antialiased=False)


fig, axes = plt.subplots(1,2,subplot_kw={"projection": "3d"},figsize=(10,5))
#             kitaev_plane(N_x,N_y,mu,t_x,scp_x, pbx = 0, pby = 0, t_y=-1,scp_y=-1 )


# Sliders and checkboxes to control the variables in interactive plot.
Nx_slider =     create_horizontal_slider(fig, [0.3, 0.1, 0.1, 0.02], r'N_x', valinit=Nx_ticks[1], valmin=Nx_ticks[0], valmax=Nx_ticks[2], valstep=Nx_ticks)
Ny_slider =     create_horizontal_slider(fig, [0.3, 0.05, 0.1, 0.02], r'N_y', valinit=Nx_ticks[1], valmin=Nx_ticks[0], valmax=Nx_ticks[2], valstep=Nx_ticks, show_ticks= True)
mu_slider =     create_vertical_slider(fig, [0.05, 0.25, 0.0225, 0.63], label=r'$\mu$',valinit=init_mu, valmin=smin, valmax=smax, valstep=sstep)
tx_slider =     create_vertical_slider(fig, [0.1, 0.25, 0.0225, 0.63], label=r'$t_x$', valinit=init_tx, valmin=smin,valmax=smax,valstep=sstep)
scpx_slider =   create_vertical_slider(fig, [0.15, 0.25, 0.0225, 0.63], label=r'$\Delta_x$', valinit=init_scpx, valmin=smin,valmax=smax,valstep=sstep, color = 'gray')
ty_slider =     create_vertical_slider(fig, [0.2, 0.25, 0.0225, 0.63], label=r'$t_y = \Delta_y$', valinit=init_tx, valmin=smin,valmax=smax,valstep=sstep, color= 'gray', active= False)
checkbox_set =  create_checkbox(fig, [0.06, 0.02, 0.16, 0.17], labels = ['Lock $\Delta=t$','PBC in x','PBC in y','Release $t_y$&$\Delta_y$'], actives = [True, False, False,False])
sync_checkbox = create_camera_sync_checkbox(fig, axes, [0.5, 0.05, 0.16, 0.2])

current_values = [Nx_slider.val, Ny_slider.val, mu_slider.val, tx_slider.val, scpx_slider.val]
current_checks = checkbox_set.get_status()

plot_on_axes(surfaces, axes, current_values + [int(b) for b in current_checks[1:3]])

# Adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.15)

def lock_scpx_handler(label):
    sync_slider(checkbox_set, tx_slider, scpx_slider, ty_slider)

checkbox_set.on_clicked(lock_scpx_handler)

def update_from_tx_handler(val):
    update_synced_sliders(checkbox_set, tx_slider, scpx_slider, ty_slider)

# `functools.partial` is only imported for bugfixing in update function
# similarly source is only used to print the widget name that prompts the update of the plot
def update(source, val):
    print(f"Update triggered by {source} due to change: {val}")
    
    global current_values
    global current_checks

    next_values = [Nx_slider.val, Ny_slider.val, mu_slider.val, tx_slider.val, scpx_slider.val]
    next_checks = checkbox_set.get_status()

    if next_values != current_values or current_checks != next_checks:
        if next_values[0:2] == current_values[0:2]:
            recreate = False
        else:
            recreate = True
        plot_on_axes(surfaces, axes, [*next_values, next_checks[1] ,next_checks[2], next_checks[3]-1, next_checks[3]-1], recreate=recreate)#ty_slider.val, ty_slider.val),N_x)
        current_values = next_values
        current_checks = next_checks
        fig.canvas.draw_idle()

tx_slider.on_changed(update_from_tx_handler)
mu_slider.on_changed(partial(update, "mu_slider"))
tx_slider.on_changed(partial(update, "tx_slider"))
scpx_slider.on_changed(partial(update, "scpx_slider"))
Nx_slider.on_changed(partial(update, "Nx_slider"))
Ny_slider.on_changed(partial(update, "Ny_slider"))
checkbox_set.on_clicked(partial(update, "checkbox_set"))

plt.show()