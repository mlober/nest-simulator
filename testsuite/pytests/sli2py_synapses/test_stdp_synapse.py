# -*- coding: utf-8 -*-
#
# test_stdp_synapse.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
A parrot_neuron that repeats the spikes from a poisson generator is
connected to an iaf_psc_alpha that is driven by inh. and exc. poisson input.
The synapse is an stdp_synapse. After the simulation, we go through the pre-
and postsyn. spike-trains spike by spike and try to reproduce the STDP
results. The final weight obtained after simulation is compared to the final
weight obtained from the test.
"""

import sys
import nest
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def resetkernel():
    """
    Reset kernel to clear reset delay and resolution parameters.
    """

    nest.ResetKernel()


### Perform a NEST Simulation with STDP synapse

# input parameters
K_exc = 8000.
K_inh = 2000.
nu = 10.
nu_x = 1.7
w_exc = 45.
w_inh = -5. * w_exc
delay = 1.0
axonal_delay = 0.0
backpr_delay = delay - axonal_delay
tau_minus = 20.

# stdp parameters
stdp_params = {}
stdp_params['alpha'] = 1.1
stdp_params['lambda'] = 0.01
stdp_params['tau_plus'] = 20.
stdp_params['mu_plus'] = 1.0
stdp_params['mu_minus'] = 1.0
stdp_params['Wmax'] = 2 * w_exc


def setup_network_and_run_simulation():

    # create
    pg_exc = nest.Create('poisson_generator', {'rate': K_exc * (nu + nu_x)})
    pg_inh = nest.Create('poisson_generator', {'rate': K_inh * nu})
    pg_pre = nest.Create('poisson_generator', {'rate': nu})

    parrot = nest.Create('parrot_neuron')
    neuron = nest.Create('iaf_psc_alpha', {'tau_minus': tau_minus})

    sr_pre = nest.Create('spike_recorder')
    sr_post = nest.Create('spike_recorder')

    # connect
    nest.SetDefaults('stdp_synapse', stdp_params)
    
    nest.Connect(pg_exc, neuron, syn_spec={'weight': w_exc, 'delay': delay})
    nest.Connect(pg_inh, neuron, syn_spec={'weight': w_inh, 'delay': delay})
    nest.Connect(pg_pre, parrot, syn_spec={'weight': w_exc, 'delay': delay})

    nest.Connect(parrot, neuron, syn_spec={'synapse_model': 'stdp_synapse', 'weight': w_exc, 'delay': delay})

    nest.Connect(parrot, sr_pre)
    nest.Connect(neuron, sr_post)


    # simulate
    nest.Simulate(10000.)

    pre_spikes = sr_pre.events['times'] + axonal_delay
    post_spikes = sr_pre.events['times'] + backpr_delay

    final_weight = nest.GetConnections(source=parrot, target=neuron).weight

    return pre_spikes, post_spikes, final_weight


### Simulate synaptic weight change via STDP without using NEST

# define helper functions

def update_K_plus(K_plus, last_pre, pre_spike):
    K_plus = 1.0 + K_plus * np.exp((last_pre - pre_spike) / stdp_params['tau_plus'])
    return K_plus


def update_K_minus(K_minus, last_post, post_spike):
    K_minus = 1.0 + K_minus * np.exp((last_post - post_spike) / tau_minus)
    return K_minus


def facilitate(w, K_plus, last_pre, post_spike):
    w + stdp_params['lambda'] * (1.0-w)**stdp_params['mu_plus'] * K_plus * np.exp((last_pre-post_spike)/stdp_params['tau_plus'])
    w = min(w, 1.0)
    return w


def depress(w, K_minus, last_post, pre_spike):
    w + stdp_params['lambda'] * stdp_params['alpha'] * w**stdp_params['mu_minus'] * K_minus * np.exp((last_post-pre_spike)/tau_minus)
    w = max(w, 0.) 
    return w 


def simulate_stdp(pre_spikes, post_spikes):
    # initial parameters
    finished = False
    K_plus = 0.
    K_minus = 0.
    last_pre = 0
    last_post = 0
    j = 0
    i = 0
    pre_spike = pre_spikes[j]
    post_spike = post_spikes[i]
    w = w_exc / stdp_params['Wmax']

    while finished == False:
        if pre_spike == post_spike:
            # pre- and post-syn. spike at the same time
            if last_post != post_spike:
                w = facilitate(w, K_plus, last_pre, post_spike)
            if last_pre != pre_spike:
                w = depress(w, K_minus, last_post, pre_spike)

            if j+1 < len(pre_spikes):
                K_plus = update_K_plus(K_plus, last_pre, pre_spike)
                j += 1
                last_pre = pre_spike
                pre_spike = pre_spikes[j]
                if i+1 < len(post_spikes):
                    K_minus = update_K_minus(K_minus, last_post, post_spike)
                    i += 1
                    last_post = post_spike
                    post_spike = post_spikes[i]
            else:
                finished = True
                exit

        else:
            if pre_spike < post_spike:
                # next spike is a pre-syn. spike
                w = depress(w, K_minus, last_post, pre_spike)
                K_plus = update_K_plus(K_plus, last_pre, pre_spike)
                if j+1 < len(pre_spikes):
                    j += 1
                    last_pre = pre_spike
                    pre_spike = pre_spikes[j]
                else:
                    finished = True
                    exit
            else:

                # next spike is a post-syn. spike
                w = facilitate(w, K_plus, last_pre, post_spike)
                K_minus = update_K_minus(K_minus, last_post, post_spike)
                if i+1 < len(post_spikes):
                    i += 1
                    last_post = post_spike
                    post_spike = post_spikes[i]
                else:
                    last_post = post_spike
                    post_spike = pre_spikes[len(pre_spikes) - 1] + nest.resolution  # to make sure we don't come here again
        
        return w


def test_weight_simulation_equals_weight_loop():
    pre_spikes, post_spikes, w_nest_simulation = setup_network_and_run_simulation()
    w_minimal_simulation = simulate_stdp(pre_spikes, post_spikes)
    w_minimal_simulation = w_minimal_simulation * stdp_params['Wmax']

    assert w_nest_simulation == pytest.approx(w_minimal_simulation, 0.5)


