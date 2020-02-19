#!/usr/env python
import numpy as np
import ttn as ttn
import torch
import ttn_tools as tt
import numpy_hamiltonians as nham
import torch_hamiltonians as tham
import scipy.sparse as spl
import scipy.sparse.linalg as sl
import time
import opt_einsum as oe
import matplotlib
import matplotlib.pyplot as plt
import pickle


def run_simulation(*args,**kwargs):
    temp_network = ttn.TreeTensorNetwork(**kwargs)
    tt.optimize_network(temp_network, *args[:-2], printf=True)
    tt.store_network(temp_network, *args[-2:])
    if temp_network.backend == 'torch':
        torch.cuda.empty_cache()
    return

if __name__ == '__main__':


    network1 = ttn.TreeTensorNetwork(system_size=16, cut=2, chilist=[256,16],
                                    hamiltonian=tham.ising_transverse_x_float32(2.5),
                                    dimension=2,tree_seed=2,backend='torch',
                                    optimize_type='greedy', ttype = torch.float32)



    # how to use run_simulation:
    # run_simulation(3, 1e-10, 100, 'test', 'hamer', system_size=16, cut=2, chilist=[256,16],
    #                                 hamiltonian=tham.ising_transverse_x_float32(2.5),
    #                                 dimension=2,tree_seed=2,backend='torch',
    #                                 optimize_type='greedy', ttype = torch.float32)
    #
    # getting corresponding energy:
    # print(sl.eigsh(tt.exact_energy(16, nham.ising_transverse_x(1.),2), which='SA')[0][0])


    test_speed = True
    if test_speed:
        tt.optimize_network(network1, 3, 1e-10, 100, printf=True)
        cors, _ = tt.mean_two_point_correlator_i_ir(network1, network1.hamiltonian[0][0], False)
        print(cors)
        print(network1.times)

        # it seems plt.show() is not working in this docker. solution:
        # https://stackoverflow.com/questions/46018102/how-can-i-use-matplotlib-pyplot-in-a-docker-container
        # plt.matshow(cors)
        # plt.savefig('test.png')
