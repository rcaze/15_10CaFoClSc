import numpy as np
#from neuron import h
import matplotlib
matplotlib.backend("PyQt5agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, \
    TextArea, HPacker

# Functions necessary to make the figure of the article
# Set the folder and the type of image I want as output
folder = "../Figs/Source/"
folder_A = "../Figs/Anim/"
folder_prez = "../../../Talks/15_05_19Paris/Figs/"
suf = ".png"


def adjust_spines(ax, spines):
    """
    removing the spines from a matplotlib graphics.
    taken from matplotlib gallery anonymous author.

    parameters
    ----------
    ax: a matplolib axes object
        handler of the object to work with
    spines: list of char
        location of the spines

    """
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            pass
            # print 'skipped'
            # spine.set_position(('outward',10)) # outward by 10 points
            # spine.set_smart_bounds(true)
        else:
            spine.set_color('none')
            # don't draw spine
            # turn off ticks where there is no spine

    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None,
                 loc=4, pad=0.1, borderpad=0.1, sep=2, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or
          prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((sizex, 0), 0, sizey, fc="none"))

        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[bars, TextArea(labely)],
                           align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False,
                                   **kwargs)


def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True,
                 **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)

    return sb


def raster(event_times_list, color='gray', colortwo='gray', fromi=None):
    """
    Creates a raster plot

    Parameters
    ----------
    event_times_list: iterable
        a list of event time iterables
    color: string
        color of vlines

    Returns
    -------
    ax: an axis containing the raster plot

    Comments
    --------
    function taken from http://scimusing.wordpress.com/. Adding a change in
    color (just bicolor for now)
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        # Change of color from a certain value
        if ith >= fromi:
            color = colortwo
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax


def epsp_measure(vrec,
                 error=0.1):
    """
    Measure EPSP peak using a heuristic,   the size of the biggest EPSP of the
    voltage trace. An EPSP is defined as the difference between the peak and
    the trough preceding it.

    Parameters
    ----------
    vrec: an array of floats
        data voltage trace
    error: a float
        the noise which need to be overcome before stopping
        this is to avoid to detect "false" through

    Returns
    -------
    Difference between peak and the trough preceding it
    """
    peak_time = np.argmax(vrec)
    peak_value = vrec[peak_time]
    # Going back from the time of the peak
    for voltage in vrec[peak_time:0:-1]:
        if voltage > peak_value + error:
            # Stop as soon as voltage is going up again
            return vrec[peak_time] - peak_value
        peak_value = voltage
    # Return the difference between the initial voltage value
    # and the peak by default
    return vrec[peak_time]-vrec[0]


def get_coordinates(sec):
    ptn = h.n3d(sec=sec)
    xc = []
    yc = []
    for i in range(int(ptn)):
        xc.append(h.x3d(i,   sec=sec))
        yc.append(h.y3d(i,   sec=sec))
    return np.array([xc,   yc])


def section_mark(coor, pos):
    """Give the coordinate of the point given the coordinates of a section
    and the position of the mark on this section"""
    # Give the distance between all the points of the section
    distances = [dist2p((coor[0, i], coor[1, i]), (coor[0, i+1], coor[1, i+1]))
                 for i in range(coor.shape[1]-1)]
    # For the first point the distance is 0
    distances = [0.] + distances

    # Give the distances of the mark
    mark_dist = max(distances) * float(pos)

    # Sort the distances
    distances = np.sort(np.array(distances))

    # Use the interpolation to obtain the coordinates of the mark
    new_x = np.interp(mark_dist, distances, coor[0])
    new_y = np.interp(mark_dist, distances, coor[1])

    return np.array([new_x,   new_y])


def line_segments(coor, nseg):
    """Segment the coordinate of all segments given the coordinates of
    the section and the number of segments
    """
    # No need for interpolation if segment and section and counfounded
    if nseg == 1:
        return [np.array(coor)]
    # Give the distance between all the points of the section
    distances = [dist2p((coor[0, i], coor[1, i]), (coor[0, i+1], coor[1, i+1]))
                 for i in range(coor.shape[1]-1)]
    # For the first point the distance is 0
    distances = [0.] + distances
    # Sort the distances
    distances = np.sort(np.array(distances,   np.float))

    # Give the distances after each segments
    seg_dists = [0.] + [(i*distances[-1]) / float(nseg)
                        for i in range(1, nseg+1)]

    # Add the points to interpolate
    total_dists = distances.tolist() + seg_dists

    # Turn into arrays and uniquify or sort
    total_dists = np.unique(np.array(total_dists,   np.float))

    # Use the interpolation to obtain the new coordinates
    new_x = np.interp(total_dists, distances, coor[0, :])
    new_y = np.interp(total_dists, distances, coor[1, :])

    new_coor = []
    for i in range(0,   nseg):
        mask1 = total_dists >= seg_dists[i]
        mask2 = total_dists <= seg_dists[i+1]
        mask = mask1 * mask2
        new_coor.append(np.array([new_x[mask],   new_y[mask]]))

    return new_coor


def get_center(pt):
    """Give the center of a section given its 3d coordinates
    """
    points = [(pt[0, i], pt[1, i]) for i in range(len(pt[0]))]
    x,   y = zip(*points)
    center = (max(x)+min(x))/2.,   (max(y) + min(y))/2.
    return center


def dist2p(x,  y):
    """Compute the distance between two points"""
    return np.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2)


def add_line(ax, lines, section, flag):
    """Add a line to a shape plot"""
    coor = get_coordinates(section)
    coor_seg = line_segments(coor, section.nseg)
    for i in range(section.nseg):
        if section.name() != "soma":
            lines[section.name()].append(ax.plot(coor_seg[i][0],
                                                 coor_seg[i][1],
                                                 '-k', color='black',
                                                 alpha=0.5,
                                                 linewidth=1)[0])

        if section.name() == "soma" and flag:
            # plot the soma as a circle
            pt = get_coordinates(section)
            center = get_center(pt)
            # get the half diameter
            rayon = 0
            for i in range(len(pt[0])):
                distance = dist2p(center, (pt[0, i], pt[1, i]))
            if distance > rayon:
                rayon = distance
            # for the model of jia need to override for the rayon
            circle = plt.Circle(center, 7, color='black', alpha=1)
            lines[section.name()].append(circle)
            # guarantee that we plot the soma only once
            flag = False
            # Add the soma
            ax.add_artist(circle)

    return lines, flag


def show_shape_syn(model, ax=None, inset=True, fill=True, alpha=1):
    """Draw the shape of a neuron and its input sites in 2D"""
    if not ax:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        add_marks = False
    else:
        add_marks = True
    # Store the different list of lines in a dictionary
    lines = {i: [] for i in model.sections.iterkeys()}
    # Flag to plot the soma only once
    flag = True
    for section in model.sections.itervalues():
        # No tracing of the model if you simply want to add marks
        if add_marks:
            break
        lines, flag = add_line(ax, lines, section, flag)

    mark_circles = []
    for i, mark in enumerate(model.marks):
        c = model.mark_color
        s = model.mark_size
        if model.mark_shape == "c":
            mark_circles.append(Circle(mark, s, color=c, linewidth=0,
                                       alpha=alpha))
        else:
            mark_circles.append(Rectangle(mark-s/2., s, s, linewidth=0,
                                          color=c, alpha=alpha))

    # Add the marks if there is a marking
    collection = PatchCollection(mark_circles, match_original=True)
    ax.add_collection(collection)

    # Set the fov
    ax.set_ylim(-100, 100)
    ax.set_xlim(-100, 100)

    adjust_spines(ax, [])
    ax.set_yticks([])
    ax.set_xticks([-100, 0, 100])
    ax.set_xlabel(r"size($\mu$m)")

    return ax, lines


def shape(model, save=None, loc_c=0):
    """Draw the shape of the model with input localisation"""
    # Plot the selected clustered input sites
    ipt_site = model.orig_locs[loc_c]
    sec = model.sections[ipt_site[0]]
    locs = np.linspace(0.5, 1, len(model.orig_locs))
    # Save the mark for futur use
    saved_marks = model.marks
    model.marks = []
    model.mark_color = "black"
    model.mark_shape = "c"
    model.mark_size = 4
    for l in locs:
        model.marks.append(section_mark(get_coordinates(sec), l))
    ax, lines = show_shape_syn(model)
    model.marks = saved_marks

    c_marks = range(7)
    c_marks.remove(loc_c)
    for c_mark in c_marks:
        # Plot the other clustered input sites
        ipt_site = model.orig_locs[c_mark]
        sec = model.sections[ipt_site[0]]
        locs = np.linspace(0.5, 1, len(model.orig_locs))
        saved_marks = model.marks
        model.marks = []
        model.mark_color = "black"
        model.mark_shape = "c"
        model.mark_size = 4
        for l in locs:
            model.marks.append(section_mark(get_coordinates(sec), l))
        ax, lines = show_shape_syn(model, ax, alpha=0.3)
        model.marks = saved_marks

    # Plot the scattered input sites
    model.mark_color = "red"
    model.mark_shape = "square"
    model.mark_size = 8
    ax, lines = show_shape_syn(model, ax, alpha=0.8)

    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=1200)
        plt.close()
    else:
        plt.show()


def vtrace(time, vrecs, n_direc, dend=None, save=None, ylim=-64, color="black"):
    """Plot the voltage trace of a compartment
    for a given stimulus (direction)"""
    fig, ax = plt.subplots(figsize=(2, 2))
    if dend is not None:
        ax.plot(time, vrecs[n_direc][dend], color=color)
    else:
        time = time[:len(vrecs[n_direc])]
        ax.plot(time, vrecs[n_direc], color=color)
    ax.set_xlim(0,  max(time))
    ax.set_ylim(ylim-16, ylim+86)
    adjust_spines(ax, [])
    ax.add_patch(Rectangle((ylim+64, ylim-16), 1000, 100,
                           linewidth=0, color="#e9e9ea"))
    plt.plot((0, 1500), (ylim, ylim), "--", color="gray", linewidth=2.1)

    if save is not None:
        plt.tight_layout()
        plt.savefig(save,   dpi=800)
        plt.close(fig)
    else:
        plt.show()

# Data visually extracted from Jia et al 2011 article
data = [[1, 0.1, 0, 0.8, 0.12, 0, 0, 0.25],
        [0.2, 1, 0.15, 0.2, 0.17, 0.25, 0.2, 0.15]]

data_h = [1, 0.8, 0.8, 0.85, 0.85, 0.78, 0.8, 0.85]


def scalebar(time, data_soma, save=None):
    """ Plot the scale bar of a voltage trace"""
    fig, ax = plt.subplots(figsize=(2, 2))
    linec = ax.plot(time, data_soma[0], color='r')
    adjust_spines(ax, [])
    add_scalebar(ax, matchx=False, matchy=False, loc=4,
                 sizex=1000, sizey=20, labelx="1s", labely="20mV")
    # Remove the plot to leave just the scale bar
    line = linec.pop(0)
    line.remove()
    del line
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=800)
        plt.close(fig)


def tuning(datas, colors=('red', 'black'), orient='N', alpha=1, save=None):
    """Draw a polar plot of the neuron tuning"""

    theta = np.arange(0, 361, 360/float(len(datas[0])))*np.pi/180

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax = plt.subplot(111,   polar=True)
    ax.set_theta_zero_location(orient)
    ax.set_theta_direction(-1)
    ax.set_rmax(np.max(datas[0]))

    for i, data in enumerate(datas):
        data = np.concatenate((data,  [data[0]]))
        ax.plot(theta, data, color=colors[i], linewidth=0.2, alpha=0.6)
        ax.fill(theta, data, color=colors[i], alpha=alpha)

    ax.grid(True)

    for item in ([ax.title,   ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(6)
        item.set_fontname("Sans")

    #  tick locations
    thetaticks = np.arange(0,  360, 45)

    #  set ticklabels location at 1.3 times the axes' radius
    ax.set_thetagrids(thetaticks,   frac=1.2)
    if save:
        plt.savefig(save, dpi=800)
        plt.close()
    else:
        plt.show()


def vtrace_syn(time, vtraces, vrest, colors=("red", "black"), save=None,
               ylim=-45):
    """Plot the voltage trace in response of different synaptic activity"""
    fig, ax = plt.subplots(figsize=(3, 1.5))
    plt.xlim(50, max(time) + 50)
    plt.ylim(vrest-1, vrest+20)
    adjust_spines(ax, ["bottom", "left"])
    plt.plot(time, vtraces[0], color=colors[0], linewidth=2)
    shift_t = 1000
    # Plot the second bigger trace
    vtr = (np.ones(shift_t) * -65).tolist() + vtraces[1].tolist()
    dt = time[1]
    rec_t = np.arange(0, time[-1] + shift_t*dt, dt)
    plt.plot(rec_t, vtr, color=colors[1], linewidth=1)

    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=1200)
        plt.close()
    else:
        plt.show()


def em(exps, meass, save=None, colors=("black", "gray", "red"), pt=[0, -1],
       ticks=None):
    """Plot an expected vs measure for different input location"""
    maxim = np.max([exps, meass])
    if ticks is None:
        xticks = [0, round((maxim+2)/2), round(maxim+2)]
        yticks = [0, round((maxim+2)/2), round(maxim+2)]
    else:
        xticks, yticks = ticks

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    for i in range(len(exps)):
        if i < len(exps) - 1:
            c = colors[1]
            a = 0.4
            mark_s = "o"
        else:
            c = colors[2]
            a = 1
            mark_s = "s"
        if i == 0:
            c = colors[0]
            a = 1
            mark_s = "o"
        # Plot the lines
        ax.plot(exps[i], meass[i], color=c, alpha=a)
        # Add marks for the first and last points.
        for j in pt:
            # Plot the selected points as big dots
            ax.scatter(exps[i][j], meass[i][j], color=c, linewidth=1,
                       marker=mark_s, alpha=a)

    ax.set_xlabel("expected (mV)")
    ax.set_ylabel("measured (mV)")
    # Set the ticks
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # Same size axis
    ax.set_xlim(0, xticks[-1])
    ax.set_ylim(0, yticks[-1])
    # Add the id line
    ax.plot(np.arange(0, maxim, 0.01), np.arange(0, maxim, 0.01), "--",
            color="black")

    adjust_spines(ax, ["bottom", "left"])

    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=900)
        plt.close()
    else:
        plt.show()


def dist_plot(result, save=None, xlab=None, xlims=[300, 700], thr=400):
    """Plot the distribution of the created depolarization"""
    fig, ax = plt.subplots(figsize=(3, 1.5))
    width = 0.9
    pref, npref = result
    mpref = float(np.max(pref))
    mnpref = float(np.max(npref))
    pref = pref / mpref
    npref = npref / mnpref
    xcoor = np.arange(len(pref))
    ax.bar(xcoor - width/2., pref,
           width, color='red', edgecolor="none")
    ax.bar(xcoor - width/2., npref,
           width, color='gray', edgecolor="none")
    if thr is not None:
        plt.axvline(x=thr, color='black', linestyle='--')

    if xlab is None:
        ax.set_xlabel(r"Number of active subunits")
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_xticks(np.arange(xlims[0]/100, xlims[1]/100)*100)
        ax.set_xticklabels(range(xlims[0]/100, xlims[1]/100+1))
    else:
        ax.set_xlabel(r"Number of active synapses")
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_xticks(np.arange(xlims[0], xlims[1], 100))
        ax.set_xticklabels(range(xlims[0], xlims[1], 100))

    ax.set_yticks([0, 0.5, 1])
    # max_f = max([max(pref), max(npref)])
    # ax.set_yticks([0, (int(max_f*0.1)*10)/2, int(max_f*0.1)*10])
    ax.set_ylabel("Freq.")
    adjust_spines(ax, ["bottom", "left"])

    if save is not None:
        plt.tight_layout()
        plt.savefig(save, dpi=800)
        plt.close()
    else:
        plt.show()


def scan_plot(result, xticks, xlab="Synaptic Bias", save=None):
    """Plot the probability of separation given a range of parameter values"""
    fig, ax = plt.subplots(figsize=(3, 1.5))
    plt.scatter(xticks, result, color="red", s=40, clip_on=False)
    plt.plot(xticks, result, color="red", linewidth=2, clip_on=False)
    ax.set_xticks([xticks[0], xticks[2], xticks[4]])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(xlab)
    ax.set_ylabel("Separability")
    adjust_spines(ax, ["bottom", "left"])

    if save is not None:
        plt.tight_layout()
        plt.savefig(save, dpi=800)
        plt.close()
    else:
        plt.show()


def fail_plot(ypoints, xpoints, xticks=None, xlab=None, save=None):
    """Plot the number of time a model is capable of separation in
    n cases of synaptic failure in our versus a linear model"""
    fig, ax = plt.subplots(figsize=(3, 1.5))
    plt.scatter(xpoints, ypoints[0], clip_on=False, s=60, color="red")
    plt.scatter(xpoints, ypoints[1], clip_on=False, s=30,
                marker="s", color="black")

    if xticks is None:
        tkx = [0.1, 0.5, 0.9]
    else:
        tkx = xticks
    ax.set_xticks(tkx)
    ax.set_xlabel(tkx)
    if xlab is None:
        ax.set_xlabel("Fraction of synaptic failure")
    else:
        ax.set_xlabel(xlab)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Separability")
    adjust_spines(ax, ["bottom", "left"])
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=800)
        plt.close()
    else:
        plt.show()


def sub_actf(theta=0.4, save=None):
    """Plot the activation function of a compartment"""
    id_line = np.arange(0, 1.1, 0.1)
    act_line = id_line * (id_line <= theta+.05) + 1 * (id_line > theta+.05)
    line_b = id_line[:(theta+0.1)*10].tolist()
    line_e = id_line[(theta+0.1)*10:].tolist()
    id_line = line_b + [theta] + line_e
    act_line = act_line.tolist() + [1]
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.plot(id_line, act_line, color="black",
             linewidth=2, clip_on=False)
    plt.plot(id_line, id_line, color="black", linewidth=2, linestyle="--")
    plt.text(theta+0.03, theta-0.06, r"$\theta$", transform=ax.transAxes,
             color="black")
    plt.axvline(theta, color="gray", linestyle="--")
    # plt.hlines(theta, 1, 0, color="gray", linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Compartment input ($\propto$mV)")
    ax.set_ylabel(r"Output ($\propto$mV)")
    ax.set_ylim(0, 1)
    adjust_spines(ax, ["bottom", "left"])
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=800)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    sub_actf(save="../Figs/Source/subf.png")
