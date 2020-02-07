import numpy as np
import torch

szi = np.array([[1., 0], [0, -1.]])
sxi = np.array([[0, 1.], [1., 0]])
syi = np.array([[0, -1.],[1., 0]])
id = np.array([[1., 0],[0, 1.]])
s_plusi = np.array([[0,1.],[0,0]])
s_minusi = np.array([[0,0],[1.,0]])


def ising_transverse_x(x,l=1.,j=1.0,):
    return [[[sxi, sxi],1, [-1.*j,-1.*j]],
                [[szi], 0, [-1.*x]], [[id], 0, [-1.*l]]]
