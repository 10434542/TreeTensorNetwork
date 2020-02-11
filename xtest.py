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

if __name__ == '__main__':

    # problem: computation time does not seem to scale linearly
    # to see the problem, run this code and see for yourself that the second loop
    # takes a lot longer to perform (why?) SOLVED: did not add cuda.synchronize()
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    network = ttn.TreeTensorNetwork(system_size=64, cut=2, chilist=[100,100],
                                    hamiltonian=tham.ising_transverse_x(3.),
                                    dimension=2,tree_seed=1, backend='torch',
                                    optimize_type='greedy')

    # testham = tham.ising_transverse_x(1.)
    # gs = tt.exact_energy(16, nham.ising_transverse_x(3.), 2)
    # a = sl.eigsh(gs, which='SA')
    # print(a[0][0])
    # a = network.root.cache_tensor
    # ti1 = time.time()
    # i = 0
    # # first loop
    # for a_network in network.root.vertical_networks:
    #     a.add_(tt.contract_network(testham[0], a_network))
    #     i+=1
    #     print('iteration %s has %smb reserved'%(i, torch.cuda.memory_reserved(device='cuda:0')/(1024**2)))
    # tf1 = time.time()
    # print(tf1-ti1)

    # uncomment below to see differences in times
    # torch.cuda.empty_cache()


    # second loop
    # i = 0]
    # c = torch.rand(1,200,200, dtype=torch.double, device='cuda:0')
    # b = torch.rand(200,200, dtype=torch.double, device='cuda:0')
    # a = torch.rand(200,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, dtype=torch.double, device='cuda:0')
    # torch.cuda.synchronize()
    # ti = time.time()
    # oe.contract('eaf,ai,ijklmnopqrstuvwxy, bjklmnopqrstuvwxy, cb, ecf -> e', c,b,a,a,b,c)
    # torch.cuda.synchronize()
    # tf = time.time()
    # print('without flattening:', tf - ti)
    # # a = torch.rand(16,2,2,2,2, dtype=torch.float, device='cuda:0')
    # torch.cuda.synchronize()
    # ti = time.time()
    # a = a.reshape(200,-1)
    # oe.contract('eaf, ai, ij, bj , cb, ecf-> e', c,b,a,a,b,c)
    # torch.cuda.synchronize()
    # tf = time.time()
    # print('with flattening:', tf - ti)

    test_speed = True
    if test_speed:
        n_range = np.arange(0,300)
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
        print(tf2-ti2)
        print(np.mean(times)*500/3600)

    # plt.scatter(n_range, times)
    # plt.xlabel(r'iteration')
    # plt.ylabel(r'seconds')
    # plt.show()
    # plt.close()
