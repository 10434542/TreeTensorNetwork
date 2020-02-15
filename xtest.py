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
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':


    network = ttn.TreeTensorNetwork(system_size=16, cut=2, chilist=[256,16],
                                    hamiltonian=tham.ising_transverse_x_float32(3.),
                                    dimension=2,tree_seed=2,
                                    optimize_type='greedy', ttype = torch.float32)
    # 3hours for 64 300,300 ising_transverse_x: 3.3682425022125244


    # uncomment below to see differences in times
    # torch.cuda.empty_cache()

    test_speed = True
    if test_speed:
        n_range = np.arange(0,10)
        times = []
        for n in n_range:
            ti2 = time.time()
            # for b_network in network.root.left.left.horizontal_networks:
            #     tt.contract_network(testham[0], b_network)
            # torch.cuda.synchronize()
            tt.optimize_tensor(network, network.root)
            tt.optimize_tensor(network, network.root.left)
            tt.optimize_tensor(network, network.root.left.left)
            tf2 = time.time()
            # torch.cuda.empty_cache()
            times.append(tf2-ti2)
            print(tt.get_energy(network, network.root))
            # print(torch.cuda.memory_reserved(device='cuda:0')/(1024**2))
            # print('iteration %s has %smb reserved'%(i, torch.cuda.memory_reserved(device='cuda:0')/(1024**2)))
        # tt.store_network('tests','ising', network)
        print(tf2-ti2)
        print(np.mean(times)*500/3600)
        rho_t, rho_r = tt.rho_bot_sites(network, [1,2])
        tt.rho_layer(network, 1)
        # tt.rho_layer(network, 1)
        # gs = tt.exact_energy(16, nham.ising_transverse_x(3.), 2)
        # a = sl.eigsh(gs, which='SA')
        # print(a[0][0])
