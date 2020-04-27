import ttn_tools as tt
from timeit import default_timer as timer
import torch
import opt_einsum as oe
import numpy as np
import pickle
import os
from torch_hamiltonians import heisenberg_nn_id_float32 as heisham
from ttn import TreeTensorNetwork, Node

sxi = torch.tensor([0])
szi = np.array([[1., 0], [0, -1.]])
sxi = np.array([[0, 1.], [1., 0]])
syi = np.array([[0, -1.],[1., 0]])

szit = torch.tensor([[1., 0], [0, -1.]], dtype=torch.double, device='cuda')
sxit = torch.tensor([[0 ,1.], [1., 0]], dtype=torch.double, device='cuda')
syit = torch.tensor([[0,-1.], [1., 0]], dtype=torch.double, device='cuda')

def simple_ham(x):
    return [[[sxit, sxit], 1, [-1.*x,-1.*x]]]

def transfer_tree_object(folder_to_foo, folder_to_store):
    if not os.path.exists(folder_to_store):
        os.makedirs(folder_to_store)

    for i in os.lisdir(folder_to_foo):
        with open(folder_to_foo+i, 'rb') as data:
            temp_dict = pickle.load(data)
            print(temp_dict['j2_used'])
    # with open(tree_object, 'rb') as data:
    #     old_file = pickle.load(data)
    #     print(old_file.file_name)

if __name__ == '__main__':
    # test_net = TreeTensorNetwork(system_size=16,
    #     cut=2, chilist=[10,10], hamiltonian=simple_ham(1.),dimension=2)
    load_network = True
    if load_network is True:
        for i in os.listdir('new_networks/2D_64_heis_v3'):
            with open('new_networks/2D_64_heis_v3/'+i, 'rb') as d3:
                print('working on ', i)
                temp_ttn_2 = pickle.load(d3)
                operators_to_use = [temp_ttn_2.hamiltonian[0][0], temp_ttn_2.hamiltonian[1][0],
                                   temp_ttn_2.hamiltonian[2][0]]
                temp_dict = {}
                if not os.path.exists('ttn_dicts'):
                    os.mkdir('ttn_dicts')
                if not os.path.exists('ttn_dicts/2D_64_heis_v3'):
                    os.mkdir('ttn_dicts/2D_64_heis_v3')
                temp_dict['single_plaquettes'] = tt.plaquette_correlators(temp_ttn_2, operators_to_use)
                with open('ttn_dicts/2D_64_heis_v3/'+temp_ttn_2.file_name, 'wb') as d4:
                    pickle.dump(temp_dict, d4)
                print('done with ', i)
                torch.cuda.empty_cache()
    transfer = False
    if transfer is True:
        # print(tt.rho_bot_sites(test_net, [1,4,5]))
        if not os.path.exists('new_networks'):
            os.mkdir('new_networks')
        if not os.path.exists('new_networks/2D_64_heis_v3'):
            os.mkdir('new_networks/2D_64_heis_v3')

        for i in os.listdir('new_dicts/2D_64_heis_v3'):
            with open('new_dicts/2D_64_heis_v3'+'/'+i, 'rb') as d:
                print('working on %s'%(i))
                temp = pickle.load(d)
                new_tensors = temp['numpy_tensors']
                temp_ttn = TreeTensorNetwork(system_size=64, cut=3, chilist=[100,100,100],
                           hamiltonian=heisham(j2=temp['j2_used']))
                for j in temp_ttn.node_list:
                    if j.layer == 0:
                        j.current_tensor = torch.cuda.FloatTensor(new_tensors[0])
                    if j.layer == 1:
                        j.current_tensor = torch.cuda.FloatTensor(new_tensors[1])
                    if j.layer == 2:
                        j.current_tensor = torch.cuda.FloatTensor(new_tensors[2])
                    if j.layer == 3:
                        j.current_tensor = torch.cuda.FloatTensor(new_tensors[3])
                new_location = 'new_networks/2D_64_heis_v3/'
                with open(new_location+temp_ttn.file_name+'.pickle', 'wb') as d2:
                    pickle.dump(temp_ttn, d2)
                    print('Network stored in %s as %s'%(new_location, temp_ttn.file_name+'.pickle'))
