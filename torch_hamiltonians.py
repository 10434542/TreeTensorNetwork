import numpy as np
import torch


# s_plusi = torch.cuda.FloatTensor([[0,1],[0,0]])
# s_minusi = torch.cuda.FloatTensor([[0,0],[1,0]])


def ising_transverse_x_float32(x,l=1.,j=1.0):
    szi = torch.cuda.FloatTensor([[1., 0], [0, -1.]])
    sxi = torch.cuda.FloatTensor([[0, 1.], [1., 0]])
    syi = torch.cuda.FloatTensor([[0, -1.],[1., 0]])
    id = torch.cuda.FloatTensor([[1., 0],[0, 1.]])

    return [[[sxi, sxi],1, [-1.*j,-1.*j]],
                [[szi], 0, [-1.*x]], [[id], 0, [-1.*l]]]


def ising_transverse_x_float64(x,l=1.,j=1.0):

    szi = torch.cuda.DoubleTensor([[1., 0], [0, -1.]])
    sxi = torch.cuda.DoubleTensor([[0, 1.], [1., 0]])
    syi = torch.cuda.DoubleTensor([[0, -1.],[1., 0]])
    id = torch.cuda.DoubleTensor([[1., 0],[0, 1.]])

    return [[[sxi, sxi],1, [-1.*j,-1.*j]],
                [[szi], 0, [-1.*x]], [[id], 0, [-1.*l]]]


def heisenberg_nn_id_float32(j2, j1=1., h=1.):

    """ Heisenberg model with j1: <i,j>; j2: <<i,j>> """
    szi = torch.cuda.FloatTensor([[1., 0], [0, -1.]])
    id = torch.cuda.FloatTensor([[1., 0],[0, 1.]])
    s_plusi = torch.cuda.FloatTensor([[0,1.],[0,0]])
    s_minusi = torch.cuda.FloatTensor([[0,0],[1.,0]])

    return [[[szi, szi], 1, [.25*j1,.25*j1]],
                [[s_minusi, s_plusi], 1, [.5*j1,.5*j1]],
                [[s_plusi, s_minusi], 1, [.5*j1,.5*j1]],
                [[s_minusi, s_plusi], 1.5, [.5*j2,.5*j2]],
                [[s_plusi, s_minusi], 1.5, [.5*j2,.5*j2]],
                [[szi, szi], 1.5, [.25*j2,.25*j2]],
                [[id], 0, [-h]]]


def heisenberg_nn_id_float64(j2, j1=1., h=1.):

    """ Heisenberg model with j1: <i,j>; j2: <<i,j>> """
    szi = torch.cuda.DoubleTensor([[1., 0], [0, -1.]])
    id = torch.cuda.DoubleTensor([[1., 0],[0, 1.]])
    s_plusi = torch.cuda.DoubleTensor([[0,1.],[0,0]])
    s_minusi = torch.cuda.DoubleTensor([[0,0],[1.,0]])

    return [[[szi, szi], 1, [.25*j1,.25*j1]],
                [[s_minusi, s_plusi], 1, [.5*j1,.5*j1]],
                [[s_plusi, s_minusi], 1, [.5*j1,.5*j1]],
                [[s_minusi, s_plusi], 1.5, [.5*j2,.5*j2]],
                [[s_plusi, s_minusi], 1.5, [.5*j2,.5*j2]],
                [[szi, szi], 1.5, [.25*j2,.25*j2]],
                [[id], 0, [-h]]]
