from traits.api import HasTraits, Int, List, Float
import numpy as np
import numpy.random as rd
import unittest
import csv
import plot
import matplotlib.pyplot as plt
from collections import deque

def bin2time(bins,
             bin_size=100):
    """
    The inverse of bin2time,   creates a list of arrays of spike
    times

    Parameters
    ----------
    bins: a numpy 2-D Boolean array
        report an event in a given bin
    bin_size: an integer
        duration of a bin in ms

    Returns
    -------
    spike_times: a list of numpy arrays
        time where an event occurs
    """
    spike_times = []
    for bin_spike_time in bins:
        spike_time = []
        for i,   bin_spike in enumerate(bin_spike_time):
            if bin_spike:
                spike_time.append(i*bin_size)
        spike_times.append(spike_time)
    return spike_times


class Stimulation(HasTraits):
    """Wrap the two functions to generate the stimulation made of three
    episodes: an evoked episode between two spontaneous episodes.

    Traits
    ------
    n_neurons: non-negative int
        number of neurons
    group_size: non-negative int
        number of correlated neurons in the evoked episode
    shift: non-negative int
        shifting the group of correlated neurons
    episode_duration: non-negative int
        duration of an episode in bin
    bin_size: non-negative int
        duration of a bin in ms (for plotting)
    spon_firing_prob: float between 0 and 1
        probability of firing in the two spontaneous episodes
    evok_firing_prob: float between 0 and 1
        probability of firing in the evoked episode
    syn : List
        determine the position of the synchronous event
    test: Bool
        Enable testing mode (useful for NEURON simulation)
    """

    n_neurons = Int(300)
    group_size = Int(100)
    group_number = Int(3)
    shift = Int(0)
    episode_duration = Int(100)
    bin_size = Int(1)
    spon_fp = Float(0.01)
    evok_fp = Float(0.01)
    sync = List([25,   50, 75])

    def _anytrait_changed(self, name, old, new):
        """Update to a change in any trait apart the one created"""
        # Skip the creation of new trait in the object
        not_these_traits = ['trait_added']
        if name in not_these_traits:
            return

        if name == 'shift':
            # Simply shift the previously generated spike train
            self.shift_spikes(new - old)
            return

        if name in ['n_neurons',   'group_size']:
            self.group_number = self.n_neurons/self.group_size

        if name in ['evok_firing_prob',   'spon_firing_prob']:
            self.spon_freq = self.spon_firing_prob / float(self.bin_size)*1000
            self.evok_freq = self.evok_firing_prob / float(self.bin_size)*1000

        if name in ['episode_duration',   'bin_size']:
            self.duration = self.episode_duration * 3
            self.duration_time = self.duration * self.bin_size

    def set_rate(self,   sync):
        """
        Determine the rate of the inhomogeneous spike train

        Parameters
        ----------
        spon_freq: float between 0 and 1
            Firing frequency during spontaneous activity
        evok_freq: float between 0 and 1
            Firing  frequency during evoked activity
        ep_duration: int
            Duration of an episode
        sync: iterable
            Determine the position of the synchronous events
        """
        ep_duration = self.episode_duration

        spon = np.ones(ep_duration,   np.float)*self.spon_fp
        evok = np.ones(ep_duration,   np.float)*self.evok_fp
        # Set the tim of the synchronous event
        for i in sync:
            evok[i] = True
        return np.concatenate((spon,   evok, spon))

    def __call__(self):
        """Generate an new set of spike trains when called"""
        try:
            assert self.group_size < self.n_neurons
        except AssertionError:
            print "reset the number of neuron to match the group size"
            self.n_neurons = self.group_size

        try:
            assert self.shift < self.n_neurons - self.group_size + 1
        except AssertionError:
            print "shift trait too high,   reset to 0"
            self.shift = 0
            return

        # Set some attribut of the object
        self.duration = self.episode_duration * 3
        self.duration_time = self.duration * self.bin_size
        self.spon_freq = self.spon_fp / float(self.bin_size)*1000
        self.evok_freq = self.evok_fp / float(self.bin_size)*1000

        # Determine if the firing rate is co-evolving or if synchronous events
        th_stim = self.set_rate(self.sync)
        th_no_stim = self.set_rate([])
        sp_stim = np.random.random((self.group_size,   len(th_stim))) <= th_stim
        sp_no_stim = np.random.random((self.n_neurons - self.group_size,
                                       len(th_stim))) <= th_no_stim
        self.spikes = np.concatenate((sp_stim,   sp_no_stim))
        return self.spikes

    def shift_spikes(self,   rotation):
        """Shifting the correlated blob
        This function does not generate spikes"""
        rotation_deque = deque(range(self.n_neurons))
        rotation_deque.rotate(rotation)
        rotation_array = np.array(rotation_deque)
        self.rotation_array = rotation_array
        self.spikes = self.spikes[rotation_array]

    def show(self, save=None):
        """Generate a figure of a raster of the input spikes"""
        fig, ax = plt.subplots(figsize=(3, 3))
        # Test if spikes where generated
        try:
            spk = self.spikes
        except AttributeError:
            self()
            spk = self.spikes

        ax = plot.raster(bin2time(spk, 10), ax)
        # ax.text(text_pos[0], text_pos[1], str(orientation))
        # Set y axis
        ax.set_ylabel("Neuron #")
        ax.set_yticks(np.array([1, self.n_neurons/2, self.n_neurons]))
        ax.set_ylim(0, self.n_neurons+0.5)
        # Set x axis
        ax.set_xlabel("Time (s)")
        ax.set_xticklabels(["0", "", "1", "", "2", "", "3"])
        plt.tick_params(which='both', direction='out')
        ax.set_xlim(0, 3000)
        # Remove top and right
        plot.adjust_spines(ax, ['bottom', 'left'])
        if save:
            plt.tight_layout()
            plt.savefig(save, dpi=800)
            plt.close(fig)
        else:
            plt.show()


class BinaryModel(HasTraits):
    """A binary neuron model where integration can be non-linear.

    Traits
    ------
    synapse: list of numpy array of int
        provide the contact to the dendritic subunit. For instance
        [[1, 2],[3,4]] means that neurons 1 and 2 are connected to dendrite 1
        and neurons 3 and 4 are connected to dendrite 2
    thresholds: list of int
        first int is the somatic threshold and the others are the threshold
        of the dendritic subunits.
    d_spike: Int
        size of the jump when a dendritic spike is occuring

    Comments
    --------
    Learning could be implemented using a third array called weights giving
    the connection strenght between a neuron and a dendritic subunit
    """
    synapses = List([[0, 2, 3], [1, 4, 5]])
    thresholds = List([1, 1, 1])
    d_spike = Int(0)

    def __call__(self,   input_spikes):
        """
        Generate the output of a Threshold Non-Linear Unit
        given a matrix of spikes trains and of synaptic_connections
        """
        # Test if synapses and threshold match with each other
        try:
            assert len(self.synapses) == len(self.thresholds) - 1
        except AssertionError:
            print "Changing thresholds trait to match synapses trait"
            self.thresholds = [1] + [1 for i in range(len(self.synapses))]
            return

        # Test if the input spikes match with the parameters
        try:
            # Find the highest neuron label in synapses
            max_connection = max([np.max(syn) for syn in self.synapses])
            assert input_spikes.shape[0] > max_connection
        except AssertionError:
            print "Not enough neurons in the population re-enter spikes"
            return
        except IndexError:
            print "Please add input spikes"
            return

        d_t = self.thresholds[1:]
        s_t = self.thresholds[0]
        dendritic_integration = np.zeros((len(self.synapses),
                                          input_spikes.shape[1]))

        for i, connections in enumerate(self.synapses):
            # Fancy indexing to threshold all the local voltage at once
            local_integration = np.sum(input_spikes[connections], axis=0)
            mask_integration = local_integration >= d_t[i]
            local_integration[mask_integration] = d_t[i] + self.d_spike
            dendritic_integration[i] = local_integration
        self.dendritic_integration = dendritic_integration
        # print np.sum(dendritic_integration,   axis=0)
        integration_result = np.sum(dendritic_integration,   axis=0)
        self.spikes = integration_result > s_t
        return self.spikes

    def show(self, dend_n):
        """
        Generate a plot of the somatic and of the dendritic activity. Possible
        only after one call of the function.
        """
        for i in range(dend_n):
            plt.plot(self.dendritic_integration[i])
        plt.show()


class TestBinaryModel(unittest.TestCase):
    def setUp(self):
        # Setup the neuron model
        self.test_model = BinaryModel()
        # Setup the spike inputs
        self.n_neurons = 16
        half_neurons = xrange(self.n_neurons/2)
        c_1 = [0 for i in half_neurons] + [1 for i in half_neurons]
        c_2 = [1 for i in half_neurons] + [0 for i in half_neurons]
        c_3 = [0 if i % 2 else 1 for i in range(self.n_neurons)]
        c_4 = [1 if i % 2 else 0 for i in range(self.n_neurons)]
        input_col = np.array([c_1,   c_2, c_3, c_4], dtype=np.bool)
        self.test_sc = np.rot90(input_col)

    def test_generate(self):
        spikes = np.ones((4, 4), dtype=np.bool)
        test = BinaryModel(spikes)
        expected_out = np.array([[2.0,   2.0, 2.0, 2.0]])
        self.assertEqual(test.dendritic_integration.tolist(),
                         expected_out.tolist())

    def test_tnlu(self):
        """Test is the binary neuron model is scatter sensitive"""
        test_model = self.test_model
        test_model.synapses = [np.arange(0,   self.n_neurons/2),
                               np.arange(self.n_neurons/2,   self.n_neurons)]
        h_n = self.n_neurons/2
        test_model.thresholds = [h_n,   h_n, 1]
        test_model.spikes = self.test_sc
        expected_out = [False, False, True, True]
        self.assertEqual(test_model.output_spikes.tolist(),   expected_out)
        expected_dend = [[h_n,   0, h_n, h_n], [0, h_n, h_n, h_n]]
        self.assertEqual(test_model.dendritic_integration.tolist(),
                         expected_dend)


def gen_fail_which(fraction, neurons_n=1000):
    """Generate all the synapses that will fail"""
    fail_frac = int(neurons_n*fraction)
    pick = rd.permutation(range(neurons_n))
    return pick[:fail_frac]


def gen_syn(dend_n, syn_n, clust, syn_loc=None):
    """Generate the connection from a neuron depending on the degree of
    Clustering

    Parameters
    ----------
    dend_n: int
        number of dendritic subunits
    syn_n: int
        number of synapses from the afferent
    clust: (float between 0 and 1, int)
        (degree of clustering, which subunit it clusters)

    Returns
    -------
    syn_loc: a list of lists of int
        synapses locations
    """
    # Number of clustered synapses
    syn_clust = int(syn_n*clust[0])

    if syn_loc is None:
        syn_loc = [[] for i in range(dend_n)]
        offset = 0
    else:
        # Find the maximum value
        offset = max([max(i) for i in syn_loc if i != []]) + 1

    for i in xrange(offset, syn_clust + offset):
        syn_loc[clust[1]].append(i)

    for i in xrange(syn_clust + offset, syn_n + offset):
        dend_ref = np.random.randint(0, dend_n)
        syn_loc[dend_ref].append(i)

    return syn_loc


def gen_arch(dend_n, syn_n, clusts):
    """Generate the synaptic architecture of the model"""
    arch = None
    for i, c_clust in enumerate(clusts):
        arch = gen_syn(dend_n, syn_n[i], c_clust, arch)
    return arch


def input_gen(syn_ns, group_n):
    """Generate the seven inputs to probe model integration"""
    ipt = np.zeros((group_n, np.sum(syn_ns)))
    ey = np.eye(group_n)

    for i, c_ipt in enumerate(ipt):
        ipt[i] = np.repeat(ey[i], syn_ns)

    return np.array(ipt, dtype=np.int)


def wrap(BIAS, SYN_N, D_THR, NDEND=7, SCAT=0, CLUST=0.3):
    """Wrap the BinaryNeuron with the input_gen and gen_arch"""
    # 8 groups of 600 synapses or less
    SYN_NS = [SYN_N] + [SYN_N - BIAS for i in range(NDEND)]
    # With a little bias in their distribution
    CLUSTS = [(SCAT, 0)] + [(CLUST, i) for i in range(NDEND)]
    model = BinaryModel()
    model.synapses = gen_arch(NDEND, SYN_NS, CLUSTS)
    model.thresholds = [1] + [D_THR for i in range(NDEND)]
    ipt = input_gen(SYN_NS, 8)
    return model, ipt.T


def results(nrep):
    """Generate the polarization distribution for preferred and non-preferred
    stimuli"""
    # Number of synapses for the prefered group
    SYN_N = 700
    # Bias between the preferred and non-preferred population
    BIAS = 50
    # Number of dendrites
    NDEND = 7
    REMOV_N = 2450
    D_THR = 100
    rec_pref = np.zeros((4, NDEND*D_THR), dtype=np.int)
    rec_npref = np.zeros((4, NDEND*D_THR), dtype=np.int)
    for i in range(nrep):
        model, ipt = wrap(BIAS, SYN_N, D_THR)
        model(ipt)
        res = np.sum(model.dendritic_integration, axis=0, dtype=np.int)
        # We -1 because indices starts at zero in python
        rec_pref[0, res[0]-1] += 1
        rec_npref[0, res[1:]-1] += 1
        # Remove half of the synapses and see how the distribution look like
        ipt_maim = np.array(ipt)
        already_pick = []
        for i in range(REMOV_N):
            zero_id = rd.randint(0, ipt_maim.shape[0])
            while zero_id in already_pick:
                zero_id = rd.randint(0, ipt_maim.shape[0])
            already_pick.append(zero_id)
            ipt_maim[zero_id] = np.zeros(ipt_maim.shape[1])
        model(ipt_maim)
        res_maim = np.sum(model.dendritic_integration, axis=0, dtype=np.int)
        rec_pref[1, res_maim[0]] += 1
        rec_npref[1, res_maim[1:]] += 1

        # Case of a linear model with a bias in the number of synapses
        saved_syn = model.synapses
        # Create a bias in the number of synapses
        # Synapses with non spatial bias.
        SYNNPREF = SYN_N - BIAS
        model.synapses = [range(0, SYN_N)] + [range(SYN_N + i*SYNNPREF,
                                                    SYN_N + (i+1)*SYNNPREF)
                                              for i in range(0, 6)]
        model.thresholds = [1] + [1000 for i in range(7)]
        model(ipt_maim)
        res3 = np.sum(model.dendritic_integration, axis=0, dtype=np.int)
        rec_pref[3, res3[0]] += 1
        rec_npref[3, res3[1:]] += 1

        model.synapses = saved_syn
        # Case where we remove half of the dendrites
        for i in range(2):
            model.synapses.pop(rd.randint(0, len(model.synapses)-1))

        # Reset the dendritic thershold at the same value
        model.thresholds = [1] + [D_THR for i in range(5)]
        model(ipt)

        res_maim2 = np.sum(model.dendritic_integration, axis=0, dtype=np.int)
        rec_pref[2, res_maim2[0]] += 1
        rec_npref[2, res_maim2[1:]] += 1

    return rec_pref, rec_npref


def results_noise(nrep, FAIL_FRACS):
    """Simulate the impact of noise on an implementation method"""
    # Number of synapses for the prefered group
    SYN_N = 700
    # Bias between the preferred and non-preferred population
    BIAS = 50
    # Number of dendrites
    D_THR = 100
    separability = np.zeros_like(FAIL_FRACS)
    separability_l = np.zeros_like(FAIL_FRACS)
    model, ipt = wrap(BIAS, SYN_N, D_THR)
    # Make the dendritic threshold so highthat integration is linear
    model_l, ipt = wrap(BIAS, SYN_N, SYN_N)
    ipt_ref = np.array(ipt)
    for i, frac in enumerate(FAIL_FRACS):
        for j in range(nrep):
            fail = gen_fail_which(frac, int(SYN_N + (SYN_N - BIAS)*7))
            ipt = np.array(ipt_ref)
            ipt[fail, :] = 0
            model(ipt)
            res = np.sum(model.dendritic_integration, axis=0, dtype=np.int)
            if res[0] > np.max(res[1:]):
                separability[i] += 1

            model_l(ipt)
            res_l = np.sum(model_l.dendritic_integration, axis=0, dtype=np.int)
            if res_l[0] > np.max(res_l[1:]):
                separability_l[i] += 1
    separability = separability / float(nrep)
    separability_l = separability_l / float(nrep)

    return separability, separability_l


def results_dend_rem(nrep, rem_dend=[1, 2, 3, 4, 5, 6]):
    """Simulate the impact of removing different number
    of dendritic compartments"""
    # Number of synapses for the prefered group
    SYN_N = 700
    # Bias between the preferred and non-preferred population
    BIAS = 50
    # dendritic threshold
    D_THR = 100
    separability = np.zeros_like(rem_dend)
    separability_l = np.zeros_like(rem_dend)
    dist_p = np.zeros(7*D_THR, dtype=np.int)
    dist_np = np.zeros(7*D_THR, dtype=np.int)
    dist_p_l = np.zeros(7*D_THR, dtype=np.int)
    dist_np_l = np.zeros(7*D_THR, dtype=np.int)

    for i, dend in enumerate(rem_dend):
        for j in range(nrep):
            model, ipt = wrap(BIAS, SYN_N, D_THR)
            model_l, ipt = wrap(BIAS, SYN_N, SYN_N)

            for i in range(dend):
                # List the synapses to remove
                del model.synapses[0]
                del model.thresholds[0]
                del model_l.synapses[0]
                del model_l.thresholds[0]

            model(ipt)
            res = np.sum(model.dendritic_integration, axis=0, dtype=np.int)
            if res[0] > np.max(res[1:]):
                separability[i] += 1

            model_l(ipt)
            res_l = np.sum(model_l.dendritic_integration, axis=0, dtype=np.int)
            if res_l[0] > np.max(res_l[1:]):
                separability_l[i] += 1

            if dend == 4:
                dist_p[res[0]] += 1
                dist_np[res[1:]] += 1
                dist_p_l[res_l[0]] += 1
                dist_np_l[res_l[1:]] += 1
        print res, res_l
    dists = [dist_p, dist_np, dist_p_l, dist_np_l]
    separability = separability / float(nrep)
    separability_l = separability_l / float(nrep)

    return separability, separability_l, dists


def scanpar(nrep, pars, par_name="clust"):
    """
    """
    SYN_N = 700
    D_THR = 100
    rec = np.zeros_like(pars, dtype=np.float)

    for i in xrange(nrep):
        for i, par in enumerate(pars):
            if par_name == "clust":
                model, ipt = wrap(0, SYN_N, D_THR, CLUST=par)
            else:
                model, ipt = wrap(par, SYN_N, D_THR)
            model(ipt)
            res = np.sum(model.dendritic_integration, axis=0, dtype=np.int)
            if res[0] > np.max(res[1:]):
                rec[i] += 1

    return rec/float(nrep)


def syn_tabl():
    """Generate the table of synaptic connection as a csv"""
    SUB_N = 7
    STIM_N = 8
    BIAS = 50
    SYN_N = 700
    SYN_TOT = 700
    DATA_FOLDER = "../Data/"
    TABL = range(STIM_N + 1)
    TOP = ["Stim. \ Comp.  "] + \
        [" " + str(i) + " " for i in range(SUB_N)] + ["Total \\\\"]
    TABL[0] = TOP
    STIM = [[" "+str(i)+" "] for i in range(0, 361, 45)]
    add = np.zeros(SUB_N)
    for i in range(1, SUB_N+2):
        if i > 1:
            SYN_N = SYN_TOT - BIAS
            add = np.zeros(SUB_N)
            add[i-2] = 0.3*SYN_N
        per_sub = SYN_N/float(SUB_N)
        end = [" " + str(SYN_N) + " \\\\"]
        if i == SUB_N + 1:
            end = [" " + str(SYN_N)]
        TABL[i] = STIM[i-1] + [" "+str(int(per_sub + add[j]))+" " for
                               j in range(SUB_N)] + end
    with open("%smodel.csv" % DATA_FOLDER, "wb") as f:
        csv.writer(f, delimiter="&").writerows(np.array(TABL))


def syn_gen():
    """Generate the distribution of synapses for the first figure"""
    D1 = range(0, 50) + range(100, 120)
    D2 = range(50, 100) + range(120, 200)
    return [D1, D2]


def plot_integration(dend, neuron, save=None, ymax=None, soma_th=False):
    """
    Plot the result of the dendritic integration of one or multiple subunit
    Parameters
    ----------
    dend: list of int giving the integration of a dendrites

    Returns
    -------
    plot of the dendritic integration
    """
    plt.close("all")

    dend_int = neuron.dendritic_integration/2.

    # Case where we want to plot somatic activity.
    if soma_th:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(np.sum(dend_int, axis=0), color="black")
        # plt.axhline(y=soma_th, linestyle='--', color='r')
        plt.xlabel("Time")
        plt.ylabel("Activity %")
        plot.adjust_spines(ax, ["left", "bottom"])
        fig.tight_layout()
        if ymax is not None:
            ax.set_ylim(0, ymax)
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()
        return


    fig, ax = plt.subplots(len(dend), 1, figsize=(4,4), sharex=True)
    for d, i in enumerate(dend):
        ax[i].plot(neuron.dendritic_integration[d], color="black")
        if ymax is not None:
            ax[i].set_ylim(0, ymax)
        plot.adjust_spines(ax[i],["left", "bottom"])
    plt.xlabel("Time")
    fig.tight_layout()
    fig.text(0., 0.55, "Activity %", va='center', rotation='vertical')

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


FIG1 = False
SUB = False
PARSCAN = False
DEND = True
DIST = False
NOISE = False

if __name__ == "__main__":
    FOLDER = "../Figs/Source/"
    SUF = ".png"
    # syn_tabl()

    if FIG1:
        stim = Stimulation()
        stim.n_neurons = 200
        stim.group_number = 2
        stim.group_size = 100

        neuron = BinaryModel()
        neuron.synapses = syn_gen()
        neuron.thresholds = [60, 40, 40]
        neuron.d_spike = 60

        stim()
        neuron(stim.spikes)
        stim.show(save="%sFig1a%s" % (FOLDER, SUF))
        plot_integration([0,1], neuron, ymax=100, save="%sFig1c%s" % (FOLDER, SUF))
        plot_integration([0,1], neuron, ymax=100, soma_th=0.6,
                         save="%sFig1e%s" % (FOLDER, SUF))


        stim.shift_spikes(100)
        neuron(stim.spikes)
        stim.show(save="%sFig1b%s" % (FOLDER, SUF))
        plot_integration([0,1], neuron, ymax=100, save="%sFig1d%s" % (FOLDER, SUF))
        plot_integration([0,1], neuron, ymax=100, soma_th=0.6,
                         save="%sFig1f%s" % (FOLDER, SUF))

    if DEND:
        NREP = 1000
        sep, sep_l, dists = results_dend_rem(NREP)
        plot.dist_plot((dists[0], dists[1]), save="%sFig3bC%s" % (FOLDER, SUF),
                       xlims=[0, 500], thr=260)
        plot.dist_plot((dists[2], dists[3]), save="%sFig3bD%s" % (FOLDER, SUF),
                       xlab="Synaptic activity",
                       xlims=[0, 500], thr=None)
        rem_dend = [1, 2, 3, 4, 5, 6]
        plot.fail_plot([sep, sep_l], rem_dend, xticks=rem_dend,
                       xlab="# of removed comparments",
                       save="%sFig3bB%s" % (FOLDER, SUF))

    if PARSCAN:
        # Vary the degree of clustering
        pars_c = [0.05, 0.07, 0.09, 0.11, 0.13]
        rclusts = scanpar(1000, pars_c)
        # Vary the bias
        pars = [-250, -225, -200, -175, -150]
        rbiass = scanpar(1000, pars, par_name="b")
        # Plot the result of the parameter scan
        plot.scan_plot(rbiass, pars, save="%sFig2A%s" % (FOLDER, SUF))
        plot.scan_plot(rclusts, pars_c, xlab="Clustering",
                       save="%sFig3A%s" % (FOLDER, SUF))
    if DIST:
        rp, rnp = results(1000)
        plot.dist_plot((rp[0], rnp[0]), save="%sFig3C%s" % (FOLDER, SUF),
                       thr=590)
        plot.dist_plot((rp[1], rnp[1]), save="%sFig3D%s" % (FOLDER, SUF),
                       thr=340)
        plot.dist_plot((rp[2], rnp[2]), save="%sFig3E%s" % (FOLDER, SUF),
                       thr=420)

    if NOISE:
        frac_fails = np.arange(0.1, 0.95, 0.1)
        frac_fails = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        NREP = 1000
        sep = results_noise(NREP, frac_fails)
        plot.fail_plot(sep, frac_fails, save="%sFig3F%s" % (FOLDER, SUF))
