import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button

def create_radio_buttons(fig, position, labels, on_clicked_callback=None, active=0):
    """
    Create a set of radio buttons in the given figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to attach the radio buttons to.
    position : list or tuple
        Rectangle [left, bottom, width, height] in figure coordinates for placement.
    labels : list of str
        Labels for each radio button option.
    on_clicked_callback : function, optional
        Callback function called with label of selected radio button on change.
    active : int, optional
        Index of the initially active radio button (default 0).

    Returns
    -------
    radio : matplotlib.widgets.RadioButtons
        The created RadioButtons widget.
    """
    ax_radio = fig.add_axes(position)
    radio = RadioButtons(ax_radio, labels)
    radio.set_active(active)
    if on_clicked_callback:
        radio.on_clicked(on_clicked_callback)
    return radio

def create_vertical_slider(fig, pos, label, valinit, valmin, valmax, valstep=None, color='black', active=True):
    """
    Create and return a vertical slider.
    pos: [left, bottom, width, height] in figure coordinates
    """
    ax = fig.add_axes(pos)
    slider = Slider(ax=ax, label=label, valmin=valmin, valmax=valmax,
                    valstep=valstep, valinit=valinit, orientation="vertical")
    slider.label.set_color(color)
    slider.set_active(active)
    if color == 'gray':
        slider.poly.set_alpha(0.1)
        slider.eventson = False
    return slider

def create_horizontal_slider(fig, pos, label, valinit, valmin, valmax, valstep=None, show_ticks = False):
    """
    Create and return a horizontal slider.
    """
    ax = fig.add_axes(pos)
    slider = Slider(ax=ax, label=label, valmin=valmin, valmax=valmax,
                    valstep=valstep, valinit=valinit, orientation="horizontal")
    if show_ticks:
        ax.add_artist(ax.xaxis)
        ax.set_xticks(valstep)
    return slider

def create_checkbox(fig, pos, labels, actives):
    """
    Create and return a CheckButtons widget.
    """
    ax = fig.add_axes(pos)
    ax.set_frame_on(False)
    checkbox = CheckButtons(ax=ax, labels=labels, actives=actives)
    return checkbox

def create_camera_sync_checkbox(fig, axes, checkbox_pos, labels=['Sync Cameras'], actives = [False]):
    """
    Creates a checkbox for camera synchronization and handles the synchronization
    of camera views between two 3D axes (plots) when the checkbox is clicked.

    Parameters:
    - fig: The figure object
    - axes: List of axes (should be 2 axes for this functionality)

    Returns:
    - checkbox: The CheckButtons widget for syncing camera views
    """
    
    # Create checkbox for camera sync
    ax_camlock = fig.add_axes(checkbox_pos)
    ax_camlock.set_frame_on(False)  
    checkbox = CheckButtons(ax=ax_camlock, labels=labels, actives=actives)

    def on_move(event):
        """Synchronize camera view between axes[0] and axes[1]."""
        if event.inaxes is axes[0]:
            axes[1].view_init(elev=axes[0].elev, azim=axes[0].azim)
        elif event.inaxes is axes[1]:
            axes[0].view_init(elev=axes[1].elev, azim=axes[1].azim)
        fig.canvas.draw()

    def camcheck(event):
        """Handles camera sync toggle when checkbox is clicked."""
        if checkbox.get_status()[0]:  # Checkbox is checked
            global concid
            concid = fig.canvas.mpl_connect('motion_notify_event', on_move)
        else:  # Checkbox is unchecked
            fig.canvas.mpl_disconnect(concid)

    checkbox.on_clicked(camcheck)
    
    # Return the checkbox object in case you want to modify or check its status later
    return checkbox

def sync_slider(checkbox_set, tx_slider, scpx_slider, ty_slider=None):
    """
    This function handles the locking and unlocking of the scpx slider based on
    the checkbox status, and updates the scpx slider value based on tx slider value.

    Parameters:
    - checkbox_set: The CheckButtons widget object to get the status.
    - scpx_slider: The slider for scpx.
    - tx_slider: The slider for tx.
    - update_from_tx: The function to call to update scpx and ty sliders when lock state changes.
    """
    if checkbox_set.get_status()[0]:
        scpx_slider.set_active(False)
        scpx_slider.eventson = False
        scpx_slider.label.set_color('gray')
        scpx_slider.poly.set_alpha(0.1)
    else:
        scpx_slider.set_active(True)
        scpx_slider.eventson = True
        scpx_slider.label.set_color('black')
        scpx_slider.poly.set_alpha(1)
    update_synced_sliders(checkbox_set, tx_slider, scpx_slider, ty_slider)

def update_synced_sliders(checkbox_set, tx_slider, scpx_slider, ty_slider=None):
    """
    This function updates the values of scpx and ty sliders based on the tx slider value
    and checkbox states.

    Parameters:
    - checkbox_set: The CheckButtons widget object to get the status.
    - tx_slider: The slider for tx.
    - scpx_slider: The slider for scpx.
    - ty_slider: The slider for ty.
    """
    if checkbox_set.get_status()[0]:  # If 'Lock scp = t' is checked
        scpx_slider.set_val(tx_slider.val)
    if ty_slider == None:   # In 2D plots there also exist another t slider for other axis, ty
        plt.draw()
    else:
        if not checkbox_set.get_status()[-1]:   # PS: Assumed ty = tx lock to be the last checkbox always
            ty_slider.set_val(tx_slider.val)
        else:
            ty_slider.set_val(0)  # Set ty to 0 when the checkbox is checked
        plt.draw()

def create_button(fig, position, label, on_click=None, hovercolor='0.975'):
    """
    Create a matplotlib Button widget.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the button to.
    position : list or tuple of 4 floats
        Position of the button as [left, bottom, width, height].
    label : str
        Text displayed on the button.
    on_click : callable
        Function to call when the button is clicked.
    hovercolor : str or float, optional
        Color of button when hovered. Default is '0.975'.

    Returns
    -------
    button : matplotlib.widgets.Button
        The created Button widget.
    """
    ax_button = fig.add_axes(position)
    button = Button(ax_button, label, hovercolor=hovercolor)
    if on_click != None:
        button.on_clicked(on_click)
    return button