
# %% Import libraries, define functions
import numpy as np
import matplotlib.pyplot as plt
from custom_functions import kitaev_chain, energy, E_k_kitaev_theory
from gui_elements import create_vertical_slider, create_checkbox, create_button, sync_slider, update_synced_sliders
#%% KITAEV CHAIN K-SPACE PLOT
#%matplotlib widget # uncomment this line if the interactive plot doesn't work as intended in notebooks
N = 20
k_vec = np.arange(0, 2, 2/N)
k_cont = np.arange(0, 2, 2/(N*10))

max_mu, max_t, max_scp =    ( 10, 10, 10 )
init_mu, init_t, init_scp = ( 0, 0, 0 )
min_mu, min_t, min_scp =    ( -10, -10, -10 )

fig, ax = plt.subplots()
line_theory, = ax.plot(k_cont, E_k_kitaev_theory(np.pi*k_cont, init_mu, init_t, init_scp)[0], lw=1)
line_calc, = ax.plot(k_vec, energy(kitaev_chain(N, init_mu, init_t, init_scp, True), apply_ft_on=0)[:,0], '.k', lw=2)
ax.set_ylim(0, 2)
ax.set_xlabel('$k/\\pi$')
ax.set_ylabel('Eigenvalue')
ax.set_title(f'Kitaev Chain N={N}')
fig.subplots_adjust(left=0.35, bottom=0.25)

# Create gui widgets using gui_elements helpers
mu_slider = create_vertical_slider(fig, [0.06, 0.25, 0.0225, 0.63], label='$\mu$', valinit=init_mu, valmin= min_mu, valmax= max_mu)
t_slider = create_vertical_slider(fig, [0.13, 0.25, 0.0225, 0.63], label='t', valinit=init_t, valmin= min_t, valmax= max_t)
scp_slider = create_vertical_slider(fig, [0.2, 0.25, 0.0225, 0.63], label='$\Delta$', valinit=init_scp, valmin= min_scp, valmax= max_scp)
checkbox_sync_scp = create_checkbox(fig, [0.1, 0.1, 0.16, 0.15], labels=['Lock $\Delta=t$'], actives=[False])
reset_button = create_button(fig, [0.8, 0.025, 0.1, 0.04], 'Reset')
zoom_in_button = create_button(fig, [0.6, 0.025, 0.13, 0.04], 'Zoom In')
zoom_out_button = create_button(fig, [0.45, 0.025, 0.13, 0.04], 'Zoom Out')

# # Handlers
# def lock_scp_handler(label):
#     if checkbox_sync_scp.get_status()[0]:
#         scp_slider.set_active(False)
#         scp_slider.label.set_color('gray')
#         scp_slider.poly.set_alpha(0.1)
#     else:
#         scp_slider.set_active(True)
#         scp_slider.label.set_color('black')
#         scp_slider.poly.set_alpha(1)
#     plt.draw()

# checkbox_sync_scp.on_clicked(lock_scp_handler)

def lock_scpx_handler(label):
    sync_slider(checkbox_sync_scp, t_slider, scp_slider)

checkbox_sync_scp.on_clicked(lock_scpx_handler)

def update(val):
    line_calc.set_ydata(energy(kitaev_chain(N, mu_slider.val, t_slider.val, scp_slider.val, True), apply_ft_on=0)[:,0])
    line_theory.set_ydata(E_k_kitaev_theory(np.pi*k_cont, mu_slider.val, t_slider.val, scp_slider.val)[0])
    fig.canvas.draw_idle()

def update_from_t_handler(val):
    update_synced_sliders(checkbox_sync_scp, t_slider, scp_slider)


def update_scp_from_t(val):
    if checkbox_sync_scp.get_status()[0]:
        scp_slider.set_val(t_slider.val)
    plt.draw()

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

mu_slider.on_changed(update)
t_slider.on_changed(update)
scp_slider.on_changed(update)
t_slider.on_changed(update_scp_from_t)

reset_button.on_clicked(reset_sliders)
zoom_in_button.on_clicked(zoom_in)
zoom_out_button.on_clicked(zoom_out)

plt.show()
