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

import nest
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def resetkernel():
    """
    Reset kernel to clear reset delay and resolution parameters.
    """

    nest.ResetKernel()

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
tau_minus = 30.

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
    nest.Connect(pg_pre, neuron, syn_spec={'weight': w_exc, 'delay': delay})

    nest.Connect(parrot, neuron, syn_spec={'synapse_model': 'stdp_synapse', 'weight': w_exc, 'delay': delay})

    nest.Connect(parrot, sr_pre)
    nest.Connect(neuron, sr_post)


    # simulate
    nest.Simulate(10000.)

    pre_spikes = sr_pre.events['times'] + axonal_delay
    post_spikes = sr_pre.events['times'] + backpr_delay

    final_weight = nest.GetConnections(source=parrot, target=neuron).weight

    return pre_spikes, post_spikes, final_weight


pre_spikes, post_spikes, final_weight = setup_network_and_run_simulation()

# initial parameters
K_plus = 0.
K_minus = 0.
last_pre = 0
last_post = 0
j = 0
i = 0

post_spike = post_spikes[i]
pre_spike = pre_spikes[j]
w = w_exc / stdp_params['Wmax']

# define functions for weight update and spike times

def update_K_plus(K_plus):
    K_plus = 1.0 + K_plus * np.exp((last_pre - pre_spike) / stdp_params['tau_plus'])
    return K_plus


def update_K_minus(K_minus):
    K_minus = 1.0 + K_minus * np.exp((last_pre - pre_spike) / stdp_params['tau_minus'])
    return K_minus


def next_post_spike():
    i += 1
    last_post = post_spike
    post_spike = post_spikes[i]


def facilitate():
    
