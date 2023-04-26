# -*- coding: utf-8 -*-
#
# test_bernoulli_synapse.py
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

import nest
import numpy as np
import pytest


class TestBernoulliSynapse:
    r"""
    Test of bernoulli_synapse.
    """

    @pytest.mark.parametrize("seed", range(123, 133))  # test 10 different rng seeds
    def test_bernoulli_synapse_statistics(self, seed: int):
        r"""Measure average number of spikes transmitted by a Bernoulli synapse (static synapse with
        stochastic transmission).

        1000 spikes generated by spike_generator are sent to a parrot neuron using a bernoulli_synapse with
        transmission probability 25%. Transmission is mediated by another parrot neuron that is connected to
        spike_generator via static synapse. If the average number of spikes detected by spike recorders is within
        three standard deviations from the mean of the binomial distribution, the synapse works fine."""

        nest.ResetKernel()
        nest.SetKernelStatus({"rng_seed": seed})

        # test parameters

        # number of spikes sent via bernoulli synapse to parrot neuron
        N_spikes = 1000

        # transmission probability of the bernoulli synapse
        p = 0.25

        # allowed absolute deviation: three standard deviations of the binomial distribution with (N_spikes, p)
        margin = 3 * (N_spikes * p * (1 - p))**.5

        # build
        sg = nest.Create("spike_generator", {"spike_times": np.arange(1, N_spikes).astype(float)})
        pre = nest.Create("parrot_neuron")
        post = nest.Create("parrot_neuron")
        sr = nest.Create("spike_recorder")

        # connect spike_generator to presynaptic parrot neuron via static synapse
        nest.Connect(sg, pre)

        # connect presynaptic parrot neuron to postsynaptic parrot neuron via bernoulli_synapse with transmission
        # probability p
        nest.Connect(pre, post, syn_spec={"synapse_model": "bernoulli_synapse",
                                          "p_transmit": p})

        # connect parrot neuron to spike_recorder
        nest.Connect(post, sr)

        # simulate for 1002 ms to allow all spikes to be recorded accordingly
        nest.Simulate(2. + N_spikes)

        # get number of spikes transmitted
        N_spikes_transmitted = len(sr.get("events")["times"])

        # mean value of spikes to be received with transmission probability p
        mean = N_spikes * p

        # check if error between number of spikes transmitted and mean is within the defined margin
        assert abs(mean - N_spikes_transmitted) <= margin

    def test_p_transmit_values(self):
        r"""Test if p_transmit values are forced to lie within the range [0, 1]"""

        nest.ResetKernel()
        pre = nest.Create("parrot_neuron")
        post = nest.Create("parrot_neuron")

        with pytest.raises(nest.kernel.NESTError):
            nest.Connect(pre, post, syn_spec={"synapse_model": "bernoulli_synapse",
                                              "p_transmit": -0.1})

        with pytest.raises(nest.kernel.NESTError):
            nest.Connect(pre, post, syn_spec={"synapse_model": "bernoulli_synapse",
                                              "p_transmit": 1.1})
