
# %% Import libraries, define functions
import numpy as np
import matplotlib.pyplot as plt
from custom_functions import kitaev_chain, eigenstates
from gui_elements import create_radio_buttons, create_vertical_slider, create_checkbox, create_button, sync_slider, update_synced_sliders
#%% KITAEV CHAIN EIGENVECTOR PLOT
#%matplotlib widget # uncomment this line if the interactive plot doesn't work as intended in notebooks
N = 20

# i have assigned the same interval for all the paramterers
max_mu, max_t, max_scp =    ( 10, 10, 10 )
init_mu, init_t, init_scp = ( 0, 0, 0 )
min_mu, min_t, min_scp =    ( -10, -10, -10 )

fig, ax = plt.subplots()
def create_plot(N=20,mu=init_mu,t=init_t,scp=init_scp):
    global line_0
    global line_1
    states = eigenstates(kitaev_chain(N, mu, t, scp))
    line_0 = ax.plot(np.sum(abs(states[:, (0, 1)]) ** 2, axis=1), 'b.-', lw=1, markersize=3)
    line_1 = ax.plot(np.sum(abs(states[:, (2, 3)]) ** 2, axis=1), 'r.-', lw=1, markersize=3)
    ax.set_xlabel('Site number')
    ax.set_ylabel('Density')
    ax.set_title(f'Open Boundary Kitaev Chain N={N}')
    ax.legend(["GS","1$^{st}$"])

create_plot(20,init_mu,init_t,init_scp)
fig.subplots_adjust(left=0.35, bottom=0.25)

cell_count_buttons = create_radio_buttons(fig, [0.8, 0.07, 0.12, 0.13], ('N=10', 'N=20', 'N=50', 'N=100'))
mu_slider = create_vertical_slider(fig, [0.06, 0.25, 0.0225, 0.63], label='$\mu$', valinit=init_mu, valmin= min_mu, valmax= max_mu)
t_slider = create_vertical_slider(fig, [0.13, 0.25, 0.0225, 0.63], label='t', valinit=init_t, valmin= min_t, valmax= max_t)
scp_slider = create_vertical_slider(fig, [0.2, 0.25, 0.0225, 0.63], label='$\Delta$', valinit=init_scp, valmin= min_scp, valmax= max_scp)
checkbox_sync_scp = create_checkbox(fig, [0.1, 0.1, 0.16, 0.15], labels=['Lock $\Delta=t$'], actives=[False])
reset_button = create_button(fig, [0.8, 0.025, 0.1, 0.04], 'Reset')
zoom_in_button = create_button(fig, [0.6, 0.025, 0.13, 0.04], 'Zoom In')
zoom_out_button = create_button(fig, [0.45, 0.025, 0.13, 0.04], 'Zoom Out')

def lock_scp_handler(label):
    sync_slider(checkbox_sync_scp, t_slider, scp_slider)

checkbox_sync_scp.on_clicked(lock_scp_handler)

def update_from_t_handler(val):
    update_synced_sliders(checkbox_sync_scp, t_slider, scp_slider)

def update_N(label):
    global N
    N = int(label[2:])
    ax.clear()
    create_plot(N, mu_slider.val, t_slider.val, scp_slider.val)
    fig.canvas.draw_idle()

def update(val):
    states = eigenstates(kitaev_chain(N,mu_slider.val,t_slider.val,scp_slider.val))
    line_0[0].set_ydata(np.sum(abs(states[:,(0,1)])**2, axis=1))
    line_1[0].set_ydata(np.sum(abs(states[:,(2,3)])**2, axis=1))
    ax.legend(["GS","1$^{st}$"])
    fig.canvas.draw_idle()

def reset_sliders(event):
    mu_slider.reset()
    t_slider.reset()
    scp_slider.reset()

def zoom_in(event):
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 0.9)
    plt.draw()

def zoom_out(event):
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.1)
    plt.draw()

# register the update function with each slider
cell_count_buttons.on_clicked(update_N)
t_slider.on_changed(update_from_t_handler)
mu_slider.on_changed(update)
t_slider.on_changed(update)
scp_slider.on_changed(update)
reset_button.on_clicked(reset_sliders)
zoom_in_button.on_clicked(zoom_in)
zoom_out_button.on_clicked(zoom_out)

plt.show()