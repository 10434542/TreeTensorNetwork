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

def heisenberg_plus_minus_id(x, j = 1.):
    """ Heisenberg model with j1: <i,j> """
    return np.array([[[szi, szi], 1, [.25*j,.25*j]],
                [[s_minusi, s_plusi], 1, [.5*j,.5*j]],
                [[s_plusi, s_minusi], 1, [.5*j,.5*j]], [[np.eye(2)], 0, [-1.]]])

def heisenberg_nn_id(j2, j1=1., h=1.):
    """ Heisenberg model with j1: <i,j>; j2: <<i,j>> """
    return [[[szi, szi], 1, [.25*j1,.25*j1]],
                [[s_minusi, s_plusi], 1, [.5*j1,.5*j1]],
                [[s_plusi, s_minusi], 1, [.5*j1,.5*j1]],
                [[s_minusi, s_plusi], 1.5, [.5*j2,.5*j2]],
                [[s_plusi, s_minusi], 1.5, [.5*j2,.5*j2]],
                [[szi, szi], 1.5, [.25*j2,.25*j2]],
                [[np.eye(2)], 0, [-h]]]
