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

if __name__ == '__main__':


    network1 = ttn.TreeTensorNetwork(system_size=16, cut=2, chilist=[256,16],
                                    hamiltonian=tham.ising_transverse_x_float32(2.5),
                                    dimension=2,tree_seed=2,backend='torch',
                                    optimize_type='greedy', ttype = torch.float32)
    # 3hours for 64 300,300 ising_transverse_x: 3.3682425022125244


    # uncomment below to see differences in times
    # torch.cuda.empty_cache()
    print(sl.eigsh(tt.exact_energy(16, nham.ising_transverse_x(1.),2), which='SA')[0][0])


    test_speed = True
    # tt.get_energy(network1, network1.root.left.left)
    if test_speed:
        tt.optimize_network(network1, 3, 1e-10, 100, printf=True)
        print(tf2-ti2)
        print(np.mean(times)*500/3600)
        cors, _ = tt.mean_two_point_correlator_i_ir(network1, network1.hamiltonian[0][0], False)
        print(cors)
    # plt.ioff()
    plt.matshow(cors)
    plt.savefig('test.png')
