COMMENT
--------------------------------------------------------------------------------
Simple synaptic mechanism derived for first order kinetics of
binding of transmitter to postsynaptic receptors.

A. Destexhe & Z. Mainen, The Salk Institute, March 12, 1993.

General references:

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  An efficient method for
   computing synaptic conductances based on a kinetic model of receptor binding
   Neural Computation 6: 10-14, 1994.  

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for
   excitable membranes, synaptic transmission and neuromodulation using a 
   common kinetic formalism, Journal of Computational Neuroscience 1: 
   195-230, 1994.
--------------------------------------------------------------------------------
During the arrival of the presynaptic spike (detected by threshold 
crossing), it is assumed that there is a brief pulse (duration=Cdur)
of neurotransmitter C in the synaptic cleft (the maximal concentration
of C is Cmax).Two state kinetic scheme synapse described by rise time tau1,
decay time constant tau2, and peak conductance gtrig.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

(Notice if tau1 -> 0 then we have just single exponential decay.)
The factor is evaluated in the
initial block such that the peak conductance is gtrig.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

Specify an incremental delivery event
(synapse starts delay after the source
crosses threshold. gtrig is incremented by the amount specified in
the delivery event, onset will be set to the proper time)

-----------------------------------------------------------------------------

KINETIC MODEL FOR GLUTAMATERGIC NMDA RECEPTORS

Whole-cell recorded postsynaptic currents mediated by NMDA receptors (Hessler
et al., Nature 366: 569-572, 1993) were used to estimate the parameters of the
present model; the fit was performed using a simplex algorithm (see Destexhe,
A., Mainen, Z.F. and Sejnowski, T.J.  Fast kinetic models for simulating AMPA,
NMDA, GABA(A) and GABA(B) receptors.  In: Computation and Neural Systems, Vol.
4, Kluwer Academic Press, in press, 1995).  The voltage-dependence of the Mg2+
block of the NMDA was modeled by an instantaneous function, assuming that Mg2+
binding was very fast (see Jahr & Stevens, J. Neurosci 10: 1830-1837, 1990;
Jahr & Stevens, J. Neurosci 10: 3178-3182, 1990).

PS: the external mg concentration is here used as a global parameter.

-----------------------------------------------------------------------------


Two state kinetic scheme synapse described by rise time tau1,
decay time constant tau2, and peak conductance gtrig.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

(Notice if tau1 -> 0 then we have just single exponential decay.)
The factor is evaluated in the
initial block such that the peak conductance is gtrig.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

Specify an incremental delivery event
(synapse starts delay after the source
crosses threshold.)
 
ENDCOMMENT

NEURON {
	POINT_PROCESS NmdaSynapse
	RANGE tau1, tau2, e, i, mgB
	GLOBAL mgo
	NONSPECIFIC_CURRENT i
	RANGE g
	GLOBAL total
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
}

PARAMETER {
	tau1=.1 (ms)
	tau2 = 10 (ms)
	e = 0	(mV)
        mgo = 1 (mM)
 	vmin = -65 (mV)
	vmax = 0 (mV)
	mg = 1    (mM)		: external magnesium concentration
}

ASSIGNED {
	v (mV)
	i (nA)
	g (umho)
	factor
	total (umho)
}

STATE {
	A (umho)
	B (umho)
	mgB	: conductance boost given by the Mg2+ block / unblock depending on the voltage
}

INITIAL {
	LOCAL tp
	total = 0
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
	mgblock(v)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	mgblock(v)
	g = B - A
	i = g * mgB * (v - e)
}

DERIVATIVE state {
	A' = -A / tau1
	B' = -B / tau2
}

PROCEDURE mgblock(v(mV)) {
	mgB = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))

}

NET_RECEIVE(weight (umho)) {
	state_discontinuity(A, A + weight*factor)
	state_discontinuity(B, B + weight*factor)
	total = total+weight
}
