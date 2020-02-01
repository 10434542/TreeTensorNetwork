import numpy as np
import torch
import opt_einsum
import cupy as cp
###############################################################################
# TOOLS FOR TTN file                                                           #
###############################################################################
torch.set_printoptions(10)

def create_sym_tensor(*dims, backend='torch'):
    """ docstring for create_sym_tensor """
    for i in dims:
        if type(i) == float:
            print(i, type(i))
            print("type is not int m8")
            raise TypeError

    if backend=='cupy':
        # print(cp.cuda.get_device_id())
        tens = cp.random.uniform(-1,1, size=[*dims])
        tens = tens+cp.transpose(tens,(0,2,1))
        tens = cp.linalg.svd(tens.reshape(dims[0], dims[1]**2), full_matrices=False)[-1]
        tens = tens.reshape(*dims)

    elif backend=='torch':
        tens = torch.rand(*dims,dtype = torch.double,device='cuda')
        tens = tens+tens.transpose(2,1)
        u, s, v = torch.svd(tens.view(dims[0], dims[1]*dims[2]), some=True)
        print(u.shape,s.shape,v.shape)
        tens = torch.svd(tens.view(dims[0], dims[1]*dims[2]), some=True)[-1]
        tens = tens.transpose(0,1).reshape(*dims)
        return tens

    elif backend=='numpy':
        tens = np.random.uniform(0,10,[*dims])
        tens = tens + np.transpose(tens, (0, 2, 1))
        tens = np.linalg.svd(tens.reshape(dims[0],dims[1]**2), full_matrices=False)[-1]
        tens = tens.reshape(*dims)

    else:
        tens=None

    return tens
