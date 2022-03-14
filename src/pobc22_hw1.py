#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor, TimedArray
from brian2 import mV, pA, ms, second, pF, Gohm
import brian2.numpy_ as np
import matplotlib.pyplot as plt


# parameters

u_rest = -65*mV
u_reset = -75*mV
u_th = -50*mV  # spike threshold
R_m = 0.02 * Gohm
C_m = 750 * pF
tau_m = R_m * C_m

t_sim = .6 * second
dt = .1 * ms


# setup brian2

brian2.defaultclock.dt = dt

# define current


def get_input_current(t_sim, dt, a_const=500*pA, a_ramp=1250*pA, a_sine=1000*pA):
    t_const = .05 * t_sim, .25 * t_sim
    t_ramp = .4 * t_sim, .6 * t_sim
    t_sine = .75 * t_sim, .85 * t_sim

    t_values = np.arange(0, t_sim, dt)
    I_values = np.zeros_like(t_values) * pA

    m = (t_const[0] <= t_values) & (t_values < t_const[1])

    # create const input current with amplitude a_const
    I_values[m] = a_const

    m = (t_ramp[0] <= t_values) & (t_values < t_ramp[1])

    # create input current ramp with peak value a_const
    I_values[m] = np.linspace(0, 1, sum(m)) * a_ramp

    m = (t_sine[0] <= t_values) & (t_values < t_sine[1])

    # create sine current (1 cycle) with peak value a_sine
    I_values[m] = np.sin(np.linspace(0, 2 * np.pi, sum(m))) * a_sine

    return t_values, I_values


# analytical response


def get_analytical_response(t_values, I_values, u_rest, u_reset, u_th, tau_m, C_m, dt):
    # assumes a spike occured at the first of the passed t_values, all
    # following spikes are taken care of automatically

    # note: assuming u_rest, u_reset, u_th, tau_m, C_m, dt are available in
    # scope

    ind0 = 0

    u_values = np.zeros_like(t_values) * mV

    while True:
        # spike occured at the time with index ind0, start integrating at
        # u_reset

        t0 = t_values[ind0]

        membrane_filter = np.exp(-(t_values[ind0:] - t0) / tau_m)

        u_values[ind0:] = u_rest + (u_reset - u_rest) * membrane_filter

        # perform convolution without dimensions and apply afterwards
        conv = np.convolve(membrane_filter, (tau_m / C_m * I_values[ind0:]) / mV, mode='full')[:len(t_values[ind0:])]
        u_values[ind0:] += conv * dt / tau_m * mV

        if (u_values[ind0:] >= u_th).any():
            ind1 = (u_values[ind0:] >= u_th).argmax()
            assert ind1 > 0
            ind0 += ind1
        else:
            break

    return t_values, u_values


# setup neuron

t_values, I_values = get_input_current(t_sim, dt)
I_t = TimedArray(I_values, dt=dt)

eqs = '''
    du / dt = (-(u - u_rest) + R_m * I_t(t)) / tau_m : volt
'''
neuron = NeuronGroup(1, eqs, threshold='u>u_th', reset='u = u_reset', method='exact')

state_mon = StateMonitor(neuron, 'u', record=0)  # monitor membrane potential of 1st neuron
spike_mon = SpikeMonitor(neuron)  # record neuron spikes

brian2.run(t_sim)

analytical_t, analytical_u = get_analytical_response(t_values, I_values, u_rest, u_reset, u_th, tau_m, C_m, dt)  # calculate analytical membrane potential

########################################################################################################################

fig, ax = plt.subplots(2)
fig.suptitle('Assignment 1 - LIF model', fontsize=16)

ax[0].plot(t_values * 1E3, I_values * 1E12, color="forestgreen", label="input current")
ax[0].set_title('Input current over time')
ax[0].legend(loc='upper left')
ax[0].set_xlabel('t / ms')
ax[0].set_ylabel('I_ext(t) / pA')

ax[1].plot(state_mon.t/ms, state_mon.u[0] * 1E3, color="mediumblue", label="Brian (simulated)")
ax[1].plot(analytical_t * 1E3, analytical_u * 1E3, linestyle='dashed', color="darkorange", label="analytical")
ax[1].set_title('Comparison of membrane potentials')
ax[1].legend(loc='upper right')
ax[1].set_xlabel('t / ms')
ax[1].set_ylabel('u(t) / mV')

# plot neuron spike
for t in spike_mon.t:
    plt.axvline(t / ms, ls='--', c='darkred', lw=3, label="neuron spike")
print('Spike times: %s' % spike_mon.t)

plt.subplots_adjust(top=0.8, hspace=0.6)
plt.show()
