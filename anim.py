""" Functions necessary for the animation """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from neuron import h
from matplotlib.pyplot import cm
from lib import NEURONModel
from plot import get_coordinates, add_line, section_mark, adjust_spines
import os
FOLDER = '../Figs/Anim/'
if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)


def show_shape(model, inset_ax=[[-82, -72], [0, 5]], select=None,
               save=None, fig_mod=None):
    """Draw the shape of a neuron and its input sites in 2D"""
    # If the function has already a figure to work with
    if fig_mod:
        ax = fig_mod.add_subplot('111')
    else:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

    lines = {i: [] for i in model.sections.iterkeys()}

    # Flat to plot the soma only once
    flag = True
    for section in model.sections.itervalues():
        lines, flag = add_line(ax, lines, section, flag)

    mark_circles = []
    for i, mark in enumerate(model.marks):
        s = model.mark_size
        c = model.mark_color
        mark_circles.append(plt.Circle(mark, s, color=c))

    # Add the marks if there is a marking
    for i in mark_circles:
        ax.add_artist(i)

    # Set fov
    ax.set_ylim(-100, 100)
    ax.set_xlim(-100, 100)
    ax.set_xticks([-100, 0, 100])
    ax.set_xlabel(r"size($\mu$m)")
    ax.set_yticks([])

    # Set the right size when making films
    if not save:
        ax.set_aspect('equal')

    if inset_ax:
        clx, cly = inset_ax
        # Creating the inset
        ax2 = fig_mod.add_axes([.75, 0.23, 0.1, 0.2])
        adjust_spines(ax2, ['left', 'bottom'])
        ax2.set_ylim(cly[0],   cly[1])
        ax2.set_ylabel("voltage(mv)")
        ax2.set_xlim(clx[0],   clx[1])
        ax2.set_xlabel("time(ms)")
        ax2.set_yticks([int(cly[0]),  int(cly[1])])
        ax2.set_xticks([0,  clx[1]])

    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=800)
        plt.close()
    else:
        if fig_mod:
            return fig_mod, ax, ax2, lines
        else:
            return ax, lines

# All global variables
time_text = None
lines = None
model = None
clim = None
cmap = None
cvals = None
ax2 = None
fig = None


def init_animation():
    """Initialize the animation"""
    global lines
    time_text.set_text('')
    return lines, time_text


def animate(i):
    """Animate at each time step"""
    for j, secname in enumerate(model.sections.iterkeys()):
        section = model.sections[secname]
        for k in range(section.nseg):
            c_n = int((cvals[secname][k][i]-clim[0])*255/(clim[1]-clim[0]))
            lines[secname][k].set_color(cmap(c_n))
    time_text.set_text('Time = %.1f ms' % model.rec_t[i])
    if i > 0:
        ax2.plot([model.rec_t[i-1], model.rec_t[i]],
                 [cvals["soma"][0][i-1], cvals["soma"][0][i]], color='black')
    return lines, time_text


def create_animation(model, nf, cl=[-80, -75], save=None):
    global ax, ax2, clim, cmap, cvals
    cvals = model.data
    cmap = cm.YlOrBr_r
    clim = cl
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=nf,
                                             init_func=init_animation,
                                             blit=True)
    if save:
        # Work best with mencoder
        ani.save(save, fps=30, bitrate=-1, writer="mencoder")
    return ani


def ani_scat_clust(model, fix, loc_c=None, vlook_f=None, video=True,
                   hyper_pol=""):
    """Produce a simple animation"""
    global fig, ax, ax2, lines, time_text
    if hyper_pol:
        clx = [-95, -10]
        cl = clx
    else:
        clx = [-65, 20]
        cl = [-65, -40]

    if loc_c is not None:
        n_stim = len(model.input_locations)
        locs = np.linspace(0.5, 1, n_stim)
        sec = model.input_locations[loc_c][0]
        new_weight = model.weights[loc_c]
        syntyp = "NmdaSynapse"
        model.input_locations = [(sec, locs[i], syntyp)
                                 for i in range(n_stim)]
        model.weights = [new_weight for i in range(n_stim)]
        coor = get_coordinates(model.sections[sec])
        model.marks = [section_mark(coor, locs[i]) for i in range(n_stim)]
        model.mark_color = "black"
        model.mark_size = 2
        distr = "Clust%s" % loc_c
    else:
        locs = model.input_locations
        model.marks = [section_mark(get_coordinates(model.sections[loc[0]]),
                                    loc[1])
                       for loc in locs]
        model.mark_color = "red"
        model.mark_size = 2
        model.mark_shape = "square"
        distr = "Scat"

    # Create and set the stimulation
    stim = np.array([[True, False] for i in range(7)])

    if hyper_pol:
        clamp = h.IClamp(model.sections["soma"](0.5))
        clamp.dur = 3000
        clamp.amp = -1

    # Launch a simulation
    model(stim, el=3)
    if vlook_f:
        return vlook(model)

    fig = plt.figure(figsize=(5, 5.5), dpi=300)

    fig, ax, ax2, lines = show_shape(model,
                                     inset_ax=[[0, int(model.rec_t[-1]/2.)],
                                               clx],
                                     fig_mod=fig)

    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    plt.savefig("%s%s%s.png" % (FOLDER, distr, hyper_pol), dpi=800)
    # Generate the anim of its first frame (as a picture to put in talks)
    if video:
        # Create the animation
        create_animation(model, int(len(model.rec_t)/2.), cl=cl,
                         save="%s%s.avi" % (FOLDER, distr))
    plt.close(fig)


def vlook(model):
    """Look at the voltage of the model
    """
    plt.close()
    plt.plot(model.rec_t, np.array(model.data["soma"][0]))
    plt.xlabel("Time (ms)")
    plt.ylabel("Volgate (mV)")
    plt.show()
    return len(model.rec_t)

if __name__ == "__main__":
    # Select fix input sites or not
    dend_num = [70, 22, 7, 42, 63, 77, 30]
    pos = [0.6, 0.66, 1.0, 1.0, 1.0, 0.6, 0.6]
    syn_type = "NmdaSynapse"
    fix = [("dend[%d]" % dend_num[i], pos[i], syn_type)
           for i in range(len(dend_num))]
    # Create the model and set the input locations
    model = NEURONModel()
    model.spike = True
    model.input_locations = fix
    model.weights = np.array([0.006 for i in range(7)])

    # Generate two animations
    ani_scat_clust(model, fix)
    ani_scat_clust(model, fix, loc_c=0)
