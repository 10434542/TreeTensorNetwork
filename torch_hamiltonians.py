import numpy as np
import torch

szi = torch.cuda.FloatTensor([[1., 0], [0, -1.]])
sxi = torch.cuda.FloatTensor([[0, 1.], [1., 0]])
syi = torch.cuda.FloatTensor([[0, -1.],[1., 0]])
id = torch.cuda.FloatTensor([[1., 0],[0, 1.]])
s_plusi = torch.cuda.FloatTensor([[0,1],[0,0]])
s_minusi = torch.cuda.FloatTensor([[0,0],[1,0]])


def ising_transverse_x(x,l=1.,j=1.0,):
    return [[[sxi, sxi],1, [-1.*j,-1.*j]],
                [[szi], 0, [-1.*x]], [[id], 0, [-1.*l]]]
