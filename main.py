import os
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import h5py
import lib
import plot
import binary


# Generate the necessary data (not necessary if lib is runned beforehand
PROJ_NAME = "14_11CaClSc"
FIG_FOLDER = "../Figs/Source/"
FIG_SUF = ".png"
DATA_FOLDER = "../Data/"

dirs = [FIG_FOLDER, DATA_FOLDER]
for c_dir in dirs:
    if not os.path.exists(c_dir):
        os.makedirs(c_dir)

try:
    hdf = h5py.File("%s/data.hdf5" % DATA_FOLDER, "r")
# Create a dataset if there is none
except IOError:
    print "Need to generate a set of data"
    model, stim = lib.generate_data()
    print "Draw model stimulated points"
    FIG_NAME = "%sFig%dD%s" % (FIG_FOLDER, 1, FIG_SUF)
    plot.shape(model, save=FIG_NAME)

# Load data into namespace
biophy = ["soma_v", "dend_v",
          "vtrace_c", "exp_c", "meas_c",
          "vtrace_s", "exp_s", "meas_s"]
hyp = ["soma_v_h", "dend_v_h", "vtrace_c_h", "exp_c_h",
       "meas_c_h", "vtrace_s_h", "exp_s_h", "meas_s_h"]
t = ["time_v", "time_em"]
data_load = biophy + hyp + t
with h5py.File("%s/data.hdf5" % DATA_FOLDER, "r") as hdf:
    for name in data_load:
        globals()[name] = np.array(hdf[name])

# Data visually extracted from Jia et al 2011 article
data = [[1, 0.1, 0, 0.8, 0.12, 0, 0, 0.25],
        [0.2, 1, 0.15, 0.2, 0.17, 0.25, 0.2, 0.15]]

data_h = [1, 0.8, 0.8, 0.85, 0.85, 0.78, 0.8, 0.85]


def fig_stim_sel(NREP=10):
    FIG_N = 2
    # Vary the bias
    pars = [-450, -300, -150, 0, 150]
    rbiass = binary.scanpar(NREP, pars, par_name="b")
    FIG_NAME = "%sFig%dA%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.scan_plot(rbiass, pars, save=FIG_NAME)

    # Plot the resistance to synaptic failure
    frac_fails = np.arange(0.1, 0.9, 0.1)
    sep = binary.results_noise(NREP, frac_fails)
    FIG_NAME = "%sFig%dB%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.fail_plot(sep, frac_fails, save=FIG_NAME)

    # Plot the result of the parameter scan
    rp, rnp = binary.results(NREP)
    FIG_NAME = "%sFig%dC%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.dist_plot((rp[0], rnp[0]), save=FIG_NAME, thr=590)
    FIG_NAME = "%sFig%dD%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.dist_plot((rp[1], rnp[1]), save=FIG_NAME, thr=340)

    # Vary the degree of clustering
    FIG_N = 3
    pars_c = [0.05, 0.07, 0.09, 0.11, 0.13]
    rclusts = binary.scanpar(NREP, pars_c)
    FIG_NAME = "%sFig%dA%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.scan_plot(rclusts, pars_c, xlab="Clustering", save=FIG_NAME)

    # Removing dendrites
    sep, sep_l, dists = binary.results_dend_rem(NREP)
    rem_dend = [1, 2, 3, 4, 5, 6]
    plot.fail_plot([sep, sep_l], rem_dend, xticks=rem_dend,
                   xlab="# of removed subunits",
                   save="%sFig3B%s" % (FIG_FOLDER, FIG_SUF))
    plot.dist_plot((dists[0], dists[1]),
                   save="%sFig3C%s" % (FIG_FOLDER, FIG_SUF),
                   xlims=[0, 500], thr=260)
    plot.dist_plot((dists[2], dists[3]),
                   save="%sFig3D%s" % (FIG_FOLDER, FIG_SUF),
                   xlab="Synaptic activity",
                   xlims=[0, 500], thr=None)


def fig_biophy():
    FIG_N = 4
    # Plot the voltage trace and the associated scale bar
    FIG_NAME = "%sFig%dA2%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.vtrace(time_v, soma_v, 0, save=FIG_NAME, color="red")
    FIG_NAME = "%sFig%dA3%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.vtrace(time_v, soma_v, 2, save=FIG_NAME, color="red")
    print FIG_NAME

    # Plot the scale bar
    FIG_NAME = "%sFig%dAScale%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.scalebar(time_v, soma_v, save=FIG_NAME)

    # Plot the tuning of the soma and dendrites
    FIG_NAME = "%sFig%dB1%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.tuning(data, save=FIG_NAME, colors=("red", "black"))
    FIG_NAME = "%sFig%dB2%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    tun = [lib.tuning_int(dend_v, 0, -65),
           lib.tuning_int(soma_v, -1, spike=True)]
    plot.tuning(tun, save=FIG_NAME, colors=("gray", "red"))

    # Plot the voltage trace for single/multiple synaptic activation
    vrest = -70
    FIG_NAME = "%sFig%dC1%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.vtrace_syn(time_em, [vtrace_s[0], vtrace_c[0][0]], vrest, save=FIG_NAME)
    FIG_NAME = "%sFig%dC2%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.vtrace_syn(time_em, [vtrace_s[3], vtrace_c[0][3]], vrest, save=FIG_NAME)

    # Plot the expected measured graph
    FIG_NAME = "%sFig%dD%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.em(exp_c.tolist() + [exp_s.tolist()],
            meas_c.tolist() + [meas_s.tolist()],
            save=FIG_NAME, ticks=[[0, 15, 30], [0, 15, 30]])


def fig_biophy_hyperpol():
    FIG_N = 5
    # Plot the dendritic trace for pref and npref
    FIG_NAME = "%sFig%dA1%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.vtrace(time_v, dend_v_h, 0, dend=0, save=FIG_NAME, color="black")
    FIG_NAME = "%sFig%dA2%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.vtrace(time_v, dend_v_h, 1, dend=0, save=FIG_NAME, color="black")

    # Plot the tuning of the soma and dendrites EXPERIMENTAL
    FIG_NAME = "%sFig%dB1%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.tuning([data_h, data[1]], colors=("red", "black"), save=FIG_NAME)

    # Plot the tuning of the soma and dendrites SIMULATION
    FIG_NAME = "%sFig%dB2%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    tun = [lib.tuning_int(soma_v_h, -1, -90), lib.tuning_int(dend_v_h, 0, -76)]
    plot.tuning([tun[0], tun[1]], colors=("red", "black"), save=FIG_NAME)

    # Plot the tuning of the same and dendrites SIMULATION
    vrest = -90
    FIG_NAME = "%sFig%dC1%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.vtrace_syn(time_em, [vtrace_s_h[0], vtrace_c_h[0][0]], vrest, save=FIG_NAME)
    FIG_NAME = "%sFig%dC2%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.vtrace_syn(time_em, [vtrace_s_h[-1], vtrace_c_h[0][-1]], vrest, save=FIG_NAME)

    # Plot the expected measured graph
    FIG_NAME = "%sFig%dD%s" % (FIG_FOLDER, FIG_N, FIG_SUF)
    plot.em(exp_c_h.tolist() + [exp_s_h.tolist()],
            meas_c_h.tolist() + [meas_s_h.tolist()],
            save=FIG_NAME, ticks=[[0, 7, 15], [0, 7, 15]])


if __name__ == "__main__":
    # fig_stim_sel(1000)
    fig_biophy()
    fig_biophy_hyperpol()
