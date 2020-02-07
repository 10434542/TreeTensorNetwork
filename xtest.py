#!/usr/env python
import numpy as np
import ttn as ttn
import torch
import ttn_tools as tt
import numpy_hamiltonians as nham
import torch_hamiltonians as tham
import time
import opt_einsum as oe

if __name__ == '__main__':

    # problem: computation time does not seem to scale linearly
    # to see the problem, run this code and see for yourself that the second loop
    # takes a lot longer to perform (why?)

    network = ttn.TreeTensorNetwork(system_size=36, cut=2, chilist=[200,200],
                                    hamiltonian=tham.ising_transverse_x(1.),
                                    dimension=2,tree_seed=1, backend='torch')

    testham = tham.ising_transverse_x(1.)
    a = network.root.cache_tensor
    ti1 = time.time()
    i = 0
    # first loop
    for a_network in network.root.vertical_networks:
        a.add_(tt.contract_network(testham[0], a_network))
        i+=1
        print('iteration %s has %smb reserved'%(i, torch.cuda.memory_reserved(device='cuda:0')/(1024**2)))
    tf1 = time.time()
    print(tf1-ti1)

    # uncomment below to see differences in times
    # torch.cuda.empty_cache()

    # second loop
    ti2 = time.time()
    i = 0
    for b_network in network.root.horizontal_networks:
        tt.contract_network(testham[0], b_network)
        i+=1
        print('iteration %s has %smb reserved'%(i, torch.cuda.memory_reserved(device='cuda:0')/(1024**2)))
    tf2 = time.time()
    print(tf2-ti2)
