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
from ttn import Node, TreeTensorNetwork, run_simulation

def main(chis):
    chis
    for chi in chis:
        run_simulation(3, 2e-8, 500, 'test_v1', 'ising_64_float_32', system_size=64, cut=2, chilist=chi,
                                        hamiltonian=tham.ising_transverse_x_float32(3.),
                                        dimension=2,tree_seed=2,backend='torch',
                                        optimize_type='greedy', ttype = torch.float32)

if __name__ == '__main__':
    main([[400,400]])
