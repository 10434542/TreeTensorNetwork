import ttn_tools as tt
from timeit import default_timer as timer
import torch
import opt_einsum as oe
import numpy as np
import pickle
import os
from torch_hamiltonians import heisenberg_nn_id_float32 as heisham
from ttn import TreeTensorNetwork, Node
import gc

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

    torch.backends.cudnn.benchmark=True
    folder_type = 'ttn_plaquettes_x'
    load_network = True
    if load_network is True:
        # for i in os.listdir('new_networks/2D_64_heis_v3'):
        with open('new_networks/2D_64_heis_v3/'+'N64_chi100-100-100_seedNone_order0.250.250.50.50.50.50.00.00.00.00.00.0-1.0.pickle', 'rb') as d3:
            # print('working on ', i)
            temp_ttn_2 = pickle.load(d3)
            operators_to_use1 = [temp_ttn_2.hamiltonian[0][0], temp_ttn_2.hamiltonian[1][0],
                               temp_ttn_2.hamiltonian[2][0]]
            temp_dict = {}
            # print(torch.cuda.memory_allocated()*(10**(-9)))
            # torch.cuda.get_device_capability()
            if not os.path.exists(folder_type):
                os.mkdir(folder_type)
            if not os.path.exists(folder_type+'/2D_64_heis_v3'):
                os.mkdir(folder_type+'/2D_64_heis_v3')

            operators_to_use2 = tt.vector_correlator(temp_ttn_2, operators_to_use1, 3)[0]

            # print(torch.__version__)
            # the print expressions below behave identical up to float32 precision
            print(tt.rho_bot_sites(temp_ttn_2,[9, 10, 18, 17, 13, 14], operators=operators_to_use2)[0].item())
            print(tt.six_point_correlator(temp_ttn_2,[9, 10, 18, 17, 13, 14], operators=operators_to_use2))

            # print(tt.rho_bot_sites(temp_ttn_2,[9, 10, 18, 17, 13, 14, 22, 21])[0])

            # print(torch.cuda.memory_allocated()*(10**(-9)))
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass
            # temp_dict['plaquettes_x'] = tt.plaquette_8(temp_ttn_2, operators_to_use)
            # with open(folder_type+'/2D_64_heis_v3/'+temp_ttn_2.file_name, 'wb') as d4:
            #     pickle.dump(temp_dict, d4)
            # print('done')
            # print('done with ', i)
            # torch.cuda.empty_cache()



    # transfer = False
    # if transfer is True:
    #     # print(tt.rho_bot_sites(test_net, [1,4,5]))
    #     if not os.path.exists('new_networks'):
    #         os.mkdir('new_networks')
    #     if not os.path.exists('new_networks/2D_64_heis_v3'):
    #         os.mkdir('new_networks/2D_64_heis_v3')
    #
    #     for i in os.listdir('new_dicts/2D_64_heis_v3'):
    #         with open('new_dicts/2D_64_heis_v3'+'/'+i, 'rb') as d:
    #             print('working on %s'%(i))
    #             temp = pickle.load(d)
    #             new_tensors = temp['numpy_tensors']
    #             temp_ttn = TreeTensorNetwork(system_size=64, cut=3, chilist=[100,100,100],
    #                        hamiltonian=heisham(j2=temp['j2_used']))
    #             t1 = torch.cuda.FloatTensor(new_tensors[0])
    #             t2 = torch.cuda.FloatTensor(new_tensors[1])
    #             t3 = torch.cuda.FloatTensor(new_tensors[2])
    #             t4 = torch.cuda.FloatTensor(new_tensors[3])
    #             for j in temp_ttn.node_list:
    #                 if j.layer == 0:
    #                     j.current_tensor = t1
    #                     j.cache_tensor = None
    #                 if j.layer == 1:
    #                     j.current_tensor = t2
    #                     j.cache_tensor = None
    #                 if j.layer == 2:
    #                     j.current_tensor = t3
    #                     j.cache_tensor = None
    #                 if j.layer == 3:
    #                     j.current_tensor = t4
    #                     j.cache_tensor = None
    #             torch.cuda.empty_cache()
    #             # does this increase ram usage? YES IT DOES, commented out for now
    #             # for j in temp_ttn.node_list:
    #             #     if j.layer == 0:
    #             #         j.current_tensor = torch.cuda.FloatTensor(new_tensors[0])
    #             #     if j.layer == 1:
    #             #         j.current_tensor = torch.cuda.FloatTensor(new_tensors[1])
    #             #     if j.layer == 2:
    #             #         j.current_tensor = torch.cuda.FloatTensor(new_tensors[2])
    #             #     if j.layer == 3:
    #             #         j.current_tensor = torch.cuda.FloatTensor(new_tensors[3])
    #             # for obj in gc.get_objects():
    #             #     try:
    #             #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             #             print(type(obj), obj.size())
    #             #     except:
    #             #         pass
    #             new_location = 'new_networks/2D_64_heis_v3/'
    #             with open(new_location+temp_ttn.file_name+'.pickle', 'wb') as d2:
    #                 pickle.dump(temp_ttn, d2)
    #                 print('Network stored in %s as %s'%(new_location, temp_ttn.file_name+'.pickle'))
    #             torch.cuda.empty_cache()
    # # # torch.cuda.empty_cache()
    # # with open('new_networks/2D_64_heis_v3/'+'N64_chi100-100-100_seedNone_order0.250.250.50.50.50.50.00.00.00.00.00.0-1.0.pickle', 'rb') as dt:
    # #     # print('working on ', i)
    # #     temp_ttn_2 = pickle.load(dt)
    # #     for j in temp_ttn_2.node_list:
    # #         print(j.value)
    # #     for obj in gc.get_objects():
    # #         try:
    # #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    # #                 print(type(obj), obj.size())
    # #         except:
    # #             pass
