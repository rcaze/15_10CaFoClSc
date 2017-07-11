import numpy as np
import matplotlib.pyplot as plt
import h5py
from neuron import h
from collections import deque
from traits.api import HasTraits, Int, Float, List, Bool
from plot import adjust_spines, raster
PROJECT_NAME = "15_01CaJaSc"
J_MDL = "/home/rcaze/Documents/Articles/15_01CaJaFoSc/Scripts/Models/Jia2011.hoc"
J_MDL_REL = "Models/Jia2011.hoc"


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

    n_neurons = Int(6)
    group_size = Int(2)
    group_number = Int(3)
    shift = Int(0)
    episode_duration = Int(100)
    bin_size = Int(1)
    spon_fp = Float(0.01)
    evok_fp = Float(0.05)
    sync = List([25,   50, 75])
    test = Bool(False)
    # In future version we can add correlation also in the spon state?
    # spon_corr = Float(0)

    def _anytrait_changed(self,   name, old, new):
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

        if self.test:
            self.spikes = np.zeros((self.n_neurons,   3), np.int)
            self.spikes[:self.group_size,   1] = 1
            return self.spikes

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
        fig, ax = plt.subplots(figsize=(3, 1.5))
        # Test if spikes where generated
        try:
            spk = self.spikes
        except AttributeError:
            self()
            spk = self.spikes

        ax = raster(bin2time(spk, 10), ax)
        # ax.text(text_pos[0], text_pos[1], str(orientation))
        # Set y axis
        # ax.set_ylabel("Neuron #")
        ax.set_yticks(np.array([1, self.n_neurons/2, self.n_neurons]))
        ax.set_ylim(0, self.n_neurons+0.5)
        # Set x axis
        # ax.set_xlabel("Time (s)")
        ax.set_xticklabels(["0", "", "1", "", "2", "", "3"])
        plt.tick_params(which='both', direction='out')
        ax.set_xlim(0, 3000)
        # Remove top and right
        adjust_spines(ax, ['bottom', 'left'])
        if save:
            plt.tight_layout()
            plt.savefig(save, dpi=800)
            plt.close(fig)
        else:
            plt.show()


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


class NEURONModel(object):
    """
    Add procedure to insert synapses and recording device on different input
    site, and control their timing
    """
    def __init__(self, hoc_model=J_MDL, dt=0.01):
        """
        Parameters
        ----------
        model: Neuron model Object
            define a model outside of an hoc file
        hoc_model: a char
            path of the file containing the neuron model
        dt: a float
            the integration timestep (fix)
        """
        # Set the temporal components
        self.dt = dt

        # Set the handlers of connection and stimulation
        self.syn, self.stim, self.vplay, self.netcon = {}, {}, {}, {}

        # Load the model contained in an hoc file
        h.load_file(1, hoc_model)
        # Add all the sections from h to be accessed from outside
        self.sections = {sec.name(): sec for sec in h.allsec()}

        # set mark parameters
        self.mark_shape = "c"
        self.mark_color = "r"
        self.mark_size = 1

    def close(self):
        """Clean up the hoc when we delete the object"""
        # This is to prevent remanescence of the synapse
        # Otherwise you can become crazy!!
        if len(self.syn) >= 1:
            for key in self.syn.keys():
                del self.syn[key]

        # Delete all the sections
        for sec in h.allsec():
            h.delete_section(sec=sec)

    def record_v(self):
        """set vectors to record the voltage in the segments of all
        sections"""
        # pre-fill all the data with soma and dendrites
        data = {i: [] for i in self.sections.iterkeys()}
        # Record the somatic voltage

        # Iterate through all the dendritic sections
        for i,   secname in enumerate(self.sections.iterkeys()):
            section = self.sections[secname]
            segments = np.linspace(0,  1, section.nseg+2)
            # iterate through all segments
            for j,   seg in enumerate(segments[1:-1]):
                # create the hoc vector for recording
                data[secname].append(h.Vector())
                # tell the vector to record the voltage
                data[secname][j].record(section(seg)._ref_v)
        self.data = data

    def fake_v(self):
        """Generate a list of voltage for each dendritic segments of the
        model"""
        self.data = {i: [np.random.normal(-65, 10, 50)]
                     for i in self.sections.iterkeys()}

    def add_stim(self, where_what, tstim, w, name='default'):
        """
        Create a synapse of a given type (Epx2Syn, NMDA, etc..) on a given
        section and activate at the time in tstim with a given intensity set by
        w. This is the where, what, when scheme. Tstim is determined by a
        vecstim object that needs to be compiled with the vecstim.mod

        Parameters
        ----------
        where_when: triple (str, float, str)
            (name of section, location between 0 and 1, type of synapse)
        name: str
            name of the synaptic object
        tstim: list
            different times at which the point process is activated
        w: float
            weight of the point process

        Comments
        --------
        IMPORTANT: This method requires the pre-compiling of vecstim.mod by
        NEURON.

        By default we insert a single Exp2Syn object. This can be moved in
        future version

        The sort command is here to make sure that tstim are in the right order.
        """
        # Select the location
        segment = self.sections[where_what[0]](where_what[1])
        # Define the type of point process to use
        meca = getattr(h,  where_what[2])
        self.syn[name] = meca(segment)
        # Try to change the time constant of the pt process
        if where_what[2] == "Exp2syn":
            self.syn[name].tau1 = 1
            self.syn[name].tau2 = 2

        if where_what[2] == "NmdaSynapse":
            self.syn[name].tau1 = 0.1
            self.syn[name].tau2 = 10

        # Convert tstim into a NEURON vector (to play in NEURON)
        self.stim[name] = h.Vector(np.sort(tstim))
        # Create play vectors to interface with NEURON
        self.vplay[name] = h.VecStim()
        # Connect vector to VecStim object to play them
        self.vplay[name].play(self.stim[name])
        # Build the netcon object to connect the stims and the synapses
        self.netcon[name] = h.NetCon(self.vplay[name],   self.syn[name])

        # Set the individual weights
        self.netcon[name].weight[0] = w

    def min_sim(self):
        """
        Launch a minimal simulation to test the model and determine its
        resting potential empirically
        """
        for sec in h.allsec():
            h.finitialize(-65, sec)
            h.fcurrent(sec)
        h.frecord_init()

        while h.t < 200:  # Launch a simulation
            h.fadvance()

        # Record the soma voltage after this minimal stimulation to find v_rest
        soma = getattr(h,   "soma")
        self.vrest = getattr(soma,   "v")

    def __call__(self,  input_spikes, bin_size=10, el=1):
        """
        Launch a simulation of a given time with a binary event happening
        or not in each time bin

        Parameters
        ----------
        input_spikes: 2-D numpy array of Bool
            contain the occurence or not of an event. Each line is a set of
            signals
            and each column corresponds to a time bin
        bin_size: float
            the size of a time bin in ms.
        el: float
            elementary weight,   it will scale all the weights

        Comments
        --------
        It seems that when multiple go method are done it does not
        change the output vector.
        """
        # Set the time to 0
        h.t = 0
        try:
            weights = self.weights*el
            where_what = self.input_locations
            assert len(self.weights) == len(self.input_locations)
        except AttributeError:
            return "Weights or the input_locations not set. Simulation aborted"
        except AssertionError:
            print "Weights or the input_locations not match. Set default w"
            self.weights = np.ones(len(self.input_locations))

        # Test if signal and stimulation match.
        try:
            assert len(input_spikes) == len(self.input_locations)
        except AssertionError:
            print self.where_what
            print input_spikes
            print "Number of input neuron does not match number of synapses"
            print "Aborting simulation"
            return

        # Set the input spike time
        self.tstims = bin2time(input_spikes,   bin_size)
        self.TSTOP = bin_size * input_spikes.shape[1]
        tstims = self.tstims

        # This is to prevent remanescence of the synapse
        # Otherwise you can become crazy!!
        if len(self.syn) >= 1:
            for key in self.syn.keys():
                del self.syn[key]

        # Set the inputs
        for i,   tstim in enumerate(tstims):
            self.add_stim(where_what[i],
                          name='Stream'+str(i),
                          tstim=tstim,
                          w=weights[i])

        # Set the initial membrane potential
        if not hasattr(self,  'vrest'):
            self.min_sim()

        # Record Time
        record_time = h.Vector()
        record_time.record(h._ref_t)

        # Record Voltage
        self.record_v()

        # Set the time at zero
        h.t = 0
        # ReInitialise the sections
        for sec in h.allsec():
            h.finitialize(self.vrest,   sec)
            h.fcurrent(sec)
        h.frecord_init()

        # Launch a simulation with or without adding spikes
        if self.spike:
            # Need to hasck the recording of the data
            self.data['soma'] = [[]]
            self.data['soma'][0].append(h.soma.v)
            refrac_abs = 200
            threshold = -40
            spk = False
            refrac = 0
            h.t = 0
            while h.t < self.TSTOP:
                h.fadvance()
                # Spike if voltage cross threshold and no refractoriness
                if h.soma.v >= threshold and not refrac:
                    h.soma.v = 20
                    spk = True

                # If the neuron spiked add one to the refractory period
                if spk:
                    refrac += 1

                # Put the soma back to a reset voltage
                if refrac == 2:
                    h.soma.v = threshold - 10

                # Reset the absolute refrac period
                if refrac >= refrac_abs:
                    refrac = 0
                    spk = False

                self.data['soma'][0].append(h.soma.v)

        else:
            while h.t < self.TSTOP:
                h.fadvance()

        # Copy time into a vector
        self.rec_t = np.array(record_time)


def synapses_location_print(neuron_model):
    """Print the location of all the synapses on a model
    the synapse location is not readable otherwise"""
    for seg in neuron_model.synapse_location:
        print seg.sec.name(),   seg.x


def tuning_int(data, dend_number, vrest=-75, spike=False):
    """Determine the tuning of a section as the integral of its voltage
    """
    # Construct the tuning of the section
    direction_number = len(data)
    tunings = np.zeros(direction_number)
    for i in range(direction_number):
        # The dend_number -1 is the soma
        if dend_number == -1:
            v = data[i]
        else:
            v = data[i][dend_number]

        # Count the number of spike to determine the tuning`
        if dend_number == -1 and spike:
            tuning = np.sum(v == 20)
        else:
            tuning = np.sum(v-vrest)

        tunings[i] = tuning

    # Normalize tuning by highest value
    tunings = tunings / float(max(tunings))

    return tunings


def set_ipt_loc_with_syn(input_sites,   syn, ipt_type="nmdanet"):
    """Determine all the input locations with the list of input sites
    fsites_numberbiophysical model"""
    try:
        assert len(syn) == len(input_sites)
    except AssertionError:
        return "#  of input sites and # of synapses groups do not match"

    nsyn = np.max(max(syn)) + 1
    # Create the set of locations
    input_locations = range(nsyn)
    for i,   cips in enumerate(input_sites):
        for csyn in syn[i]:
            input_locations[csyn] = cips

    return input_locations


def set_syn_thr(n_dend, group_size, clust_sens=False, offset=0):
    """Generate the set of synaptic connection and dendritic thresholds
    so that the neuron model is scatter sensitive. This the simplest condition
    for which it is happening.

    Parameters
    ----------
    n_dend: int
        number of dendritic subunits
    group_size: int
        number of neurons in a group. There are n_dend+1 groups
        so in total there are group_size*(n_dend+1) neurons
    clust_sens: bool
        say if the disposition of synapses is for a cluster or
        scatter sensitive neuron
    offset: int
        addition to the somatic spike to avoid spurious firing

    Returns
    -------
    synapses: list of list of int
        synaptic contact made on each input site
    threshold: list of int
        threshold of the subunits and the soma

    Examples
    --------
    """
    if clust_sens:
        n_neurons = (n_dend + 1) * group_size
        synapses = np.arange(n_neurons)
        synapses = np.array_split(synapses,   n_dend)
        synapses = [i.tolist() for i in synapses]
        threshold = [group_size + 2,  group_size] + [group_size + 1
                                                     for i in range(n_dend-1)]
        return synapses,   threshold

    # Set the number of dendrites and the number of correlated neurons
    unshared = range(n_dend)
    # Split as evenly as possible the shared inputs between dendrites
    shared = np.array_split(range(group_size),   n_dend)
    for i in range(n_dend):
        # Set the not shared connections
        unshared[i] = range((i+1)*group_size,   (i+2)*group_size)
        # Set the shared connections
        shared[i] = shared[i].tolist()
    synapses = [shared[i] + unshared[i] for i in range(n_dend)]
    thresholds = [group_size - 1 + offset] + [1 + offset for i in range(n_dend)]
    return synapses,   thresholds


def section_print():
    """Printing the different section in the NEURON namespace"""
    for sec in h.allsec():
        print sec.name()


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


def show_v(model,   sec_name, seg_num=0, color="blue"):
        """
        Show the voltage trace after a single simulation
        """
        ax = plt.gca()
        time = model.rec_t
        try:
            voltage = np.array(model.data[sec_name][seg_num])
            plt.plot(time,   voltage, color=color)
        except AttributeError:
            print "No simulation have been launched"
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_xlim(0,   model.TSTOP)
        ax.tight_layout()
        return ax


def expected_measured(model, strength, loc_c=None, rec_loc=("soma", 0),
                      bin_size=100):
        """Compute the expected versus measure epsp using a method a la Polsky.
        It make a single run where it stimulate each segmet individualy and
        then one segment

        Parameters
        ----------
        model: NEURONModel
            The model we are analysing
        strength: 1-D array
            The different strenght of the synaptic weights (1 should create a
            0.1mV depolarization at the soma)
        clustered_loc: int
            The number of the location where inputs are clustered
        """
        # Set the recording for the model
        model.record_v()

        # Test of there is an attribut original_location
        try:
            model.orig_locs
        except AttributeError:
            model.orig_locs = model.input_locations

        # Use the original location of the inputs (not demultiplied)
        n_input = len(model.orig_locs)
        # Save the demultiplied locations
        save_locs = model.input_locations
        save_w = model.weights
        # Select only all locations one time
        model.input_locations = model.orig_locs
        model.weights = model.orig_weights
        # Select only one location to do the EM
        if loc_c is not None:
            pos = model.orig_locs[loc_c]
            sec = pos[0]
            n_stim = range(len(model.orig_locs))
            locs = np.linspace(0.5, 1, len(model.orig_locs))
            model.input_locations = [(sec, locs[i], pos[2]) for i in n_stim]
            model.weights = np.array([model.weights[loc_c] for i in n_stim])
            n_input = len(model.input_locations)
        # Set the input
        matrix_input = np.concatenate((np.zeros((n_input, 1)),
                                       np.eye(n_input),
                                       np.ones((n_input, 1))), axis=1)

        vtraces = []
        expected = []
        measured = []

        for w in strength:
            model(matrix_input, bin_size=bin_size, el=w)
            # Record from the first segment of the soma
            vtrace = np.array(model.data[rec_loc[0]][rec_loc[1]])
            vtraces.append(vtrace)
            # Split the vtrace corresponding to the different episodes
            vtrace_seg = np.array_split(vtrace,   n_input+2)
            epsps = []
            for i in range(1,   n_input+1):
                epsps.append(epsp_measure(vtrace_seg[i]))
            epspglob = epsp_measure(vtrace_seg[n_input+1])

            # Add the max value of the epsp
            expected += [np.sum(epsps)]
            measured += [epspglob]

        # Restoring the demultiplied weights and locs
        if hasattr(model, "orig_locs"):
            model.weights = save_w
            model.input_locations = save_locs

        return vtraces, expected, measured


def synaptic_democracy(model, step, target_epsp, max_w=5):
    """Change the set of weights so that each location produce the same
    depolarization at the soma
    """
    model.record_v()
    # Starting from a weight of one
    locations = model.input_locations

    for i,   loc in enumerate(locations):
        # Set all the stimulation
        input_spikes = [[False] for k in range(len(locations))]
        input_spikes[i] = [True]
        # Put initial value of the weight to zero
        model.weights[i] = 0
        epsp = 0
        while model.weights[i] < max_w:
            # Increase one weight of the step value
            model.weights[i] += step
            # Run a simulation
            model(np.array(input_spikes))
            epsp = epsp_measure(np.array(model.data["soma"][0]))
            # Stop if the epsp is superior to the target epsp
            if epsp > target_epsp:
                # Break the while loop when the target epsp is reached
                break
        if model.weights[i] == max_w:
            print "end up because of weights"

    return model.weights


def gen_primary_paths(model):
    """Generate all the possible primary path from the soma"""
    # Initialise with all the child from the soma
    pps = [[i.name()] for i in h.SectionRef(sec=model.sections["soma"]).child]
    for sec_name in pps:
        # Obtain all the child for the first section in the path
        child = h.SectionRef(sec=model.sections[sec_name[0]]).child
        child_names = [sec.name() for sec in child]
        sec_name += child_names
        i = 1
        # Go through the all the section in the path
        while i < len(sec_name):
            child = h.SectionRef(sec=model.sections[sec_name[i]]).child
            child_names = [sec.name() for sec in child]
            sec_name += child_names
            i += 1

    return pps


def get_coordinates(sec):
    ptn = h.n3d(sec=sec)
    xc = []
    yc = []
    for i in range(int(ptn)):
        xc.append(h.x3d(i,   sec=sec))
        yc.append(h.y3d(i,   sec=sec))
    return np.array([xc,   yc])


def dist2p(x,  y):
    """Compute the distance between two points"""
    return np.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2)


def section_mark(coor, pos):
    """Give the coordinate of the point given the section and the location
    of the mark"""
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


def gen_locations(model, nloc, syn_type="nmdanet", fix=None):
    """Generate the different synapses randomly"""
    # Generate a list of position from which to pick from randomly
    position = np.linspace(0.5,   1, nloc)
    # List all the section name and remove soma
    dendrites = [sec_name for sec_name in model.sections.iterkeys()]
    dendrites.remove("soma")
    # Make a list of all the sections
    sections = model.sections.copy()
    sections.pop("soma")
    # Initialize the values for the output
    dend = []
    pos = []
    # Remember the coordinates of all the input_sites
    marks = []
    # Need to initialize distance
    h.distance()
    primary_paths = gen_primary_paths(model)

    if fix is not None:
        # Compute the position of the marks
        marks = []
        for f_c in fix:
            coor = get_coordinates(model.sections[f_c[0]])
            marks.append(section_mark(coor, f_c[1]))
        return fix, marks

    for i in xrange(nloc):
        # Set a flag to determine if the position is correctly chosen
        ok = False
        while not ok:
            sec_name = np.random.choice(dendrites)
            sec = sections[sec_name]
            p = np.random.choice(position)
            # Get the coordinates of the section to know if it is in the fov
            coor = get_coordinates(sec)
            mark = section_mark(coor, p)

            # Compute distance to make sure the input_location location is ok
            dist = h.distance(p, sec=sec)
            if dist > 60 and dist < 80:
                ok = True
            else:
                ok = False

        # This guarantee that each input site is on a different primary path
        # Select the primary path on which is the section
        for path in primary_paths:
            if sec_name in path:
                primary_path = path
        # Remove the all the possibly selected sections on the same path
        for sec_names in primary_path:
            dendrites.remove(sec_names)
        # Records the positions of the locations
        pos.append(p)
        dend.append(sec_name)
        marks.append(mark)

    # Create the scattered connections for the first group
    input_sites = []
    for i, d in enumerate(dend):
        input_sites.append((d, pos[i], syn_type))

    return input_sites, marks


def set_simul(sites_number=7, group_size=10, democracy=True,
              syn_type="NmdaSynapse", fix=None):
    """Set the model and stimulation object to be run"""
    try:
        assert sites_number <= group_size
    except AssertionError:
        return "The number of group should be higher than the group size"
        #  Otherwise bug in the weights that need to be fixed
    model = NEURONModel()
    # The model is spiking
    model.spike = True
    # Pick random locations as input sites for the model
    input_locations, marks = gen_locations(model, sites_number,
                                           syn_type=syn_type,
                                           fix=fix)
    model.input_locations = input_locations
    model.marks = marks

    model.weights = np.ones(len(model.input_locations))
    # Set the input weights
    if democracy:
        if syn_type == "Exp2Syn":
            model.weights = synaptic_democracy(model, target_epsp=0.5,
                                               step=0.0001, max_w=5)
        else:
            model.weights = synaptic_democracy(model, target_epsp=0.5,
                                               step=0.001, max_w=5)
    else:
        model.weights = np.ones(sites_number)
    # Save the locations and weights before demultiplication
    model.orig_weights = model.weights
    model.orig_locs = model.input_locations
    # Create a Stim object and set it
    stim = Stimulation()
    stim.n_neurons = (sites_number+1)*group_size
    stim.group_size = group_size
    # Demultiply the number of inputs sites of the model
    syn,   thr = set_syn_thr(len(model.input_locations), stim.group_size)
    model.input_locations = set_ipt_loc_with_syn(model.input_locations,   syn)
    # Demultiply the weights
    ref_table = {(loc[0], loc[1]): model.orig_weights[i]
                 for i, loc in enumerate(model.orig_locs)}
    model.weights = np.array([ref_table[(loc[0],  loc[1])]
                              for loc in model.input_locations])
    return model,   stim


def run_simul(model,  stim, el=None, crop=None, bin_size=1):
    """Run a set of n simulations with the different inputs to fill the data,
    suppose that the model as the right amount of input_locations"""
    # Scale the weights given the group_size and the number of input sites
    # if not el:
    #     el = len(model.orig_locs) / float(stim.group_size)
    # Generate a set of spikes for stim
    stim.shift = 0
    stim()

    # Create the range of shift
    shift_range = np.linspace(0,   stim.n_neurons, stim.group_number,
                              endpoint=False)
    # Run stim once to have spikes
    # Initiate the list that will contain the data
    data_soma = []
    data_dend = []
    # Crop the shift range to accelerate the simulations
    if crop:
        shift_range = shift_range[:crop]
    # Run the simulations
    for i,   shift in enumerate(shift_range):
        # Shift the group of correlated neurons
        stim.shift = int(shift)
        print "progression:",   i/float(len(shift_range))
        # Select only the stimulation episode
        spikes = stim.spikes[:,  100:250]
        model(spikes,   bin_size=bin_size, el=el)
        # Record the data for the soma
        data_soma.append(np.array(model.data['soma'][0]))
        # Record the data for all the input sites on dendrites
        temp_dend = []
        for cloc in model.orig_locs:
            sec_name = cloc[0]
            sec = model.sections[sec_name]
            loc = int(sec.nseg*cloc[1]) - 1
            dat = np.array(model.data[sec_name][loc],   np.float)
            temp_dend.append(dat)
        data_dend.append(temp_dend)
    return data_soma, data_dend


def vlook(model):
    """Look at the voltage of the model
    """
    plt.close()
    plt.plot(model.rec_t,   np.array(model.data["soma"][0]))
    plt.xlabel("Time (ms)")
    plt.ylabel("Volgate (mV)")
    plt.show()
    return len(model.rec_t)


def generate_data(short=False):
    """Generate all the data"""
    DATA_FOLDER = "../Data/"
    EM = 7
    TUNING = 8
    if short:
        EM = 2
        TUNING = 2
    sites_n = 7
    group_size = 35
    # Select fix input sites or not
    dend_num = [70, 22, 7, 42, 63, 77, 30]
    pos = [0.6, 0.66, 1.0, 1.0, 1.0, 0.6, 0.6]
    syn_type = "NmdaSynapse"
    fix = [("dend[%d]" % dend_num[i], pos[i], syn_type)
           for i in range(len(dend_num))]
    model, stim = set_simul(sites_n, group_size, syn_type=syn_type, fix=fix)
    stim.spon_fp = 0.01
    stim.evok_fp = 0.1
    stim.sync = np.random.randint(0,  100, 20).tolist()
    bin_size = 10
    # Lower the elementary weights because the baseline is higher
    el = 0.28
    print "Generating tuning data in control condition"
    soma_v, dend_v = run_simul(model, stim, el,
                               bin_size=bin_size, crop=TUNING)
    time_v = np.array(model.rec_t)
    data_savelist = ["soma_v", "dend_v"]

    print "Generating expected/measured data in control condition"
    # Cancel spiking
    model.spike = False
    strength = np.arange(1, 6)*el*2
    exp_c = range(EM)
    meas_c = range(EM)
    vtrace_c = range(EM)
    for i in range(EM):
        print "site:", i
        vtrace_c[i], exp_c[i], meas_c[i] = expected_measured(model,
                                                             strength,
                                                             i)
    vtrace_s, exp_s, meas_s = expected_measured(model, strength)
    data_savelist += ["vtrace_s", "exp_s", "meas_s",
                      "vtrace_c", "exp_c", "meas_c"]
    print "Generating data in hyperpolarized condition"
    model.spike = True
    # Inject current in the cell
    clamp = h.IClamp(model.sections["soma"](0.5))
    clamp.dur = 3000
    clamp.amp = -0.2
    baseline = -100
    model.min_sim()
    soma_v_h, dend_v_h = run_simul(model, stim, el,
                                   bin_size=bin_size, crop=TUNING)
    data_savelist += ["soma_v_h", "dend_v_h", "time_v"]

    print "Generating expected/measured data in hyperpolarized condition"
    # Generate the data for only one site
    EM = 7
    exp_c_h = range(EM)
    meas_c_h = range(EM)
    vtrace_c_h = range(EM)
    strength = np.arange(1, 6)*el
    for i in range(EM):
        print "site:", i
        vtrace_c_h[i], exp_c_h[i], meas_c_h[i] = expected_measured(model,
                                                                   strength,
                                                                   i)
    vtrace_s_h, exp_s_h, meas_s_h = expected_measured(model, strength)
    time_em = np.array(model.rec_t)
    data_savelist += ["vtrace_s_h", "exp_s_h", "meas_s_h",
                      "vtrace_c_h", "exp_c_h", "meas_c_h", "time_em"]
    print "Saving data"
    with h5py.File("%s/data.hdf5" % DATA_FOLDER, "w") as hdf:
        for name in data_savelist:
            ar = locals()[name]
            hdf.create_dataset(name, data=ar)

    return model, stim

if __name__ == "__main__":
    generate_data()
