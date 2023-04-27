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

# stdp parameters
stdp_params = {}
stdp_params['alpha'] = 1.1
stdp_params['lambda'] = 0.01
stdp_params['tau_plus'] = 20.
stdp_params['tau_minus'] = 30.
stdp_params['mu_plus'] = 1.0
stdp_params['mu_minus'] = 1.0
stdp_params['Wmax'] = 2 * w_exc