import numpy as np
import torch
import time
import opt_einsum as oe
import scipy.sparse as spl
import scipy.sparse.linalg as sl
import pickle
import os
# import cupy as cp
################################################################################
# TOOLS FOR TTN file                                                           #
################################################################################

# TODO: Entire file: add docstrings

torch.set_printoptions(10)

def timer(func):
    """ Wrapper for store_time method of class TreeData """
    @functools.wraps(func)
    def f(*args, **kwargs):
        before = time.time()
        rv = func(*args, **kwargs)
        after = time.time()
        args[0].store_time(after - before)
        # print(after-before)
        return rv
    return f

def store_network(folder_to_store, ham_name, tree_object):
    """ Stores network in folder 'stored_networks/hamiltonian' where hamiltonian can differ
    per class instance of Do_experiment'"""
    temp_folder = folder_to_store
    temp_folder += '/' + ham_name
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    network_folder = temp_folder + '/'
    if not os.path.exists(network_folder):
        os.makedirs(network_folder)

    file_name = tree_object.file_name+'.pickle'
    file_name = temp_folder+'/'+file_name

    with open(file_name, 'wb') as data:
        pickle.dump(tree_object, data)
    print('Network stored in %s as %s'%(network_folder, tree_object.file_name+'.pickle'))

def load_network(folder_to_check, ham_name, network_name, print_load = True):
    if type(network_name) != str:
        raise TypeError('path is not of type string')

    elif not os.path.exists(folder_to_check+'/'+ham_name):
        raise FileNotFoundError('No folder %s found '%(ham_name))
    elif os.path.exists(folder_to_check+'/'+ham_name) and not os.path.exists(folder_to_check+'/'+ham_name+'/'+network_name):
        raise FileNotFoundError('No file %s found in folder %s'%(network_name, ham_name))
    else:
        path_to_network = folder_to_check+'/'+ham_name+'/'+network_name
        with open(path_to_network, 'rb') as data:
            tree_object = pickle.load(data)
        if print_load:
            print('Network %s loaded'%(tree_object.file_name))
        return tree_object

def create_cache_tensor(*dims, ttype, backend='torch'):
    for i in dims:
        if type(i) == float:
            print(i, type(i))
            print("type is not int m8")
            raise TypeError
    if backend == 'torch':
        tens = torch.zeros(*dims, dtype=ttype, device='cuda:0')
    elif backend == 'numpy':
        tens = np.zeros(dims)
    return tens

def create_tensor(*dims, ttype, backend = 'torch'):
    for i in dims:
        if type(i) == float:
            print(i, type(i))
            print("type is not int m8")
            raise TypeError
    if backend == 'torch':
        tens = torch.rand(*dims, dtype=ttype, device='cuda:0')
        tens = tens.T.svd(some=True)[0]
        tens = tens.T.reshape(*dims)
    elif backend == 'numpy':
        tens = np.random.rand(*dims)
        tens = np.linalg.svd(tens, full_matrices = False)[-1]
    return tens

def create_sym_tensor(*dims, ttype, backend='torch'):
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
        tens = torch.ones(*dims, dtype = ttype, device='cuda:0').random_(0,1)
        # tens = torch.cuda.FloatTensor(*dims).random_(0, 1)
        tens.add_(tens.transpose(2,1))
        # transpose is need, for explanation see:
        # https://github.com/pytorch/pytorch/issues/24900
        tens = tens.reshape(dims[0],dims[1]*dims[1]).T
        u, s, v = tens.svd(some=True)
        tens = u.T.reshape(*dims)
        return tens

    elif backend=='numpy':
        tens = np.random.uniform(0,10,[*dims], dtype=ttype)
        tens = tens + np.transpose(tens, (0, 2, 1))
        tens = np.linalg.svd(tens.reshape(dims[0],dims[1]**2), full_matrices=False)[-1]
        tens = tens.reshape(*dims)

    else:
        tens=None

    return tens

def get_absolute_distance(lattice, spacing, constraints):
    # TODO: write this function as complementary function to get_bonds
    # then rewrite get_bonds in terms of this function
    """ docstring for get_absolute_distance() """
    pass

def get_bonds(lattice, sub_lattice, spacings):
    """ docstring for get_bonds """

    for space in spacings:
        linear_size = lattice.shape[0]
        left_boundaries, lower_boundaries = [], []
        vertical_inner_bonds, horizontal_inner_bonds = [], []
        single_sites = []
        for i in sub_lattice.flatten():
            locations = np.where(lattice==i)
            m, n = *locations[0], *locations[1]
            original_location = lattice[m,n]
            if (lattice[m,(n-space)%linear_size] not in sub_lattice) and (space>0):
                left_boundaries.append([original_location, lattice[m,(n-space)%linear_size]][::-1])
            if (lattice[(m+space)%linear_size, n] not in sub_lattice) and(space>0):
                lower_boundaries.append([original_location, lattice[(m+space)%linear_size, n]][::-1])
            if space>0:
                horizontal_inner_bonds.append([original_location, lattice[m, (n+space)%linear_size]])
                vertical_inner_bonds.append([original_location, lattice[(m-space)%linear_size,n]])
            single_sites.append([original_location])

    return (horizontal_inner_bonds, vertical_inner_bonds, single_sites, lower_boundaries,
        left_boundaries)

def get_single_network(list_of_nodes, bond):
    """ Docstring for get_single_network """
    temporary_network = []
    if len(bond) == 2:
        for a_node in list_of_nodes:
            if any(a_bond[1] == bond[1] for a_bond in a_node.vertical_two_site_terms):
                temporary_network.append(a_node)
            if any(a_bond[1] == bond[1] for a_bond in a_node.horizontal_two_site_terms):
                temporary_network.append(a_node)
            if any(a_bond[1] == bond[1] for a_bond in a_node.vertical_bc_terms):
                temporary_network.append(a_node)
            if any(a_bond[1] == bond[1] for a_bond in a_node.horizontal_bc_terms):
                temporary_network.append(a_node)
    if len(bond) == 1:
        for a_node in list_of_nodes:
            if any(a_bond[0] == bond[0] for a_bond in a_node.one_site_terms):
                temporary_network.append(a_node)

    temporary_network = list(set(temporary_network))
    temporary_network.sort(key=lambda x: x.layer)
    return {'bond':  bond, 'temporary_network': temporary_network}

def get_legs(cut, node, network):
    """ Docstring for get_legs() """

    for current_network in network:
        bond = current_network['bond']
        tensors_to_loop_over = current_network['temporary_network']
        # maybe add sort of tensors_to_loop_over over here, first value,
        # then layer
        max_leg = None
        environment_legs = None
        operator_legs = []
        all_legs = []
        current_value = node.value
        tensors_to_loop_over.sort(key = lambda x: x.layer)

        for current_node in tensors_to_loop_over:
            if current_node.isRoot():
                current_node.bralegs = np.array([1, 2, 3])
                current_node.ketlegs = np.array([1, None, None])
                if current_node.left in tensors_to_loop_over and not current_node.right in tensors_to_loop_over:
                    current_node.ketlegs[2] = current_node.bralegs[2]
                elif current_node.right in tensors_to_loop_over and not current_node.left in tensors_to_loop_over:
                    current_node.ketlegs[1] = current_node.bralegs[1]

                mask_legs = np.where(current_node.ketlegs == None)[0]
                new_values = np.arange(np.max(current_node.bralegs)+1,
                    np.max(current_node.bralegs)+mask_legs.size+1)
                current_node.ketlegs[mask_legs] = new_values
                max_leg =  np.max(np.array([np.max(current_node.ketlegs), np.max(current_node.bralegs)]))

            if not current_node.isRoot():
                # print(current_node.value, current_node.current_tensor)
                current_node.bralegs = [None]*len(current_node.current_tensor.shape)
                current_node.ketlegs = [None]*len(current_node.current_tensor.shape)

                current_node.bralegs, current_node.ketlegs = np.array(current_node.bralegs), np.array(current_node.ketlegs)
                if current_node.isLeftChild():
                    current_node.bralegs[0] = current_node.parent.bralegs[1]
                    current_node.ketlegs[0] = current_node.parent.ketlegs[1]
                if current_node.isRightChild():
                    current_node.bralegs[0] = current_node.parent.bralegs[2]
                    current_node.ketlegs[0] = current_node.parent.ketlegs[2]
                if current_node.layer != cut:
                    mask_legs = np.where(current_node.bralegs == None)[0]
                    new_bralegs = np.arange(max_leg+1, max_leg+mask_legs.size+1)
                    current_node.bralegs[mask_legs] = new_bralegs
                    max_leg = np.max(current_node.bralegs)

                # below is for lower legs of a node
                if current_node.left in tensors_to_loop_over and current_node.right in tensors_to_loop_over:
                    mask_legs = np.where(current_node.ketlegs == None)[0]
                    current_node.ketlegs[mask_legs] = np.arange(max_leg+1, max_leg+mask_legs.size+1)
                    max_leg = np.max(current_node.ketlegs)

                elif current_node.left in tensors_to_loop_over and not current_node.right in tensors_to_loop_over:
                    current_node.ketlegs[2] = current_node.bralegs[2]
                    current_node.ketlegs[1] = max_leg+1
                    max_leg = np.max(current_node.ketlegs)

                elif current_node.right in tensors_to_loop_over and not current_node.left in tensors_to_loop_over:
                    current_node.ketlegs[1] = current_node.bralegs[1]
                    current_node.ketlegs[2] = max_leg+1
                    max_leg = np.max(current_node.ketlegs)

                elif not current_node.left in tensors_to_loop_over and not current_node.right in tensors_to_loop_over:
                    current_node.bralegs = np.concatenate(([current_node.bralegs[0]], np.arange(max_leg+1, max_leg+len(current_node.current_tensor.shape))))
                    max_leg = np.max(current_node.bralegs)
                    current_node.ketlegs = np.concatenate(([current_node.ketlegs[0]], current_node.bralegs[1:]))

                    for j in bond:
                        mask_site = np.where(current_node.lattice.flatten() == j)[0]+1
                        # print(mask_site)
                        if mask_site.size>0:
                            current_node.ketlegs[mask_site] = max_leg+1
                            operator_legs.append(np.array([current_node.bralegs[mask_site][0], max_leg+1]))
                            max_leg = np.max(current_node.ketlegs)

                try:
                    max_leg = np.max(np.array([np.max(current_node.ketlegs), np.max(current_node.bralegs)]))
                except TypeError:
                    print('Whoopsie, you tried to operate on a NoneType. Pls stahp')

            if current_node.value == current_value:
                environment_legs = current_node.bralegs
            all_legs.extend((current_node.bralegs,current_node.ketlegs))
            # if node.isRoot():
            #     print(operator_legs, bond)

        # if reverse_bool:
        #     operator_legs = operator_legs[::-1]
        all_legs.extend(operator_legs[:])
        copy_shape = []
        current_network['full_legs'] = all_legs
        for i in all_legs:
            copy_shape.append(i.shape[0])

        copylegs = np.hstack(all_legs)
        open_legs = np.arange(1, len(current_node.current_tensor.shape)+1, dtype = 'int')*-1

        for i, env_leg_i in enumerate(environment_legs):
            new_open_legs_mask = np.where(copylegs == env_leg_i)[0]
            copylegs[new_open_legs_mask] = open_legs[i]

        new_closed_legs_mask = np.where(copylegs>0)[0]
        unique_closed_legs = np.unique(copylegs[new_closed_legs_mask])
        new_closed_legs = np.arange(1, unique_closed_legs.size+1)

        for i, unique_leg_i in enumerate(unique_closed_legs):
            temp_mask =np.where(copylegs ==unique_leg_i)
            copylegs[temp_mask] = new_closed_legs[i]

        tracking = [0]
        new_environment_legs = []
        for i in copy_shape:
            tracking.append(tracking[-1]+i)

        # i.all() does not work properly, use not (i< 0).all() < 0 instead
        new_environment_legs = [i for i in np.array_split(copylegs, tracking) if len(i)>0 and not (i< 0).all()]
        full_tree, environment_tree = [], []

        # manual sorting
        for i in tensors_to_loop_over:
            full_tree.extend((i,i))
            if i.value == current_value:
                environment_tree.append(i)
            else:
                environment_tree.extend((i,i))

        current_network['entire_network'] = full_tree
        current_network['unique_tensors'] = tensors_to_loop_over
        current_network['environment'] = environment_tree
        current_network['environment_legs'] = new_environment_legs

def get_optimal_order(node, dict_of_networks, optimize_type):
    """ Docstring for get_optimal_orders() """
    copied_environment_legs = [np.copy(l) for l in
                               dict_of_networks['environment_legs']]
    list_of_tensors, new_path = [], []
    new_path_energy = []
    copied_energy_legs = np.copy(dict_of_networks['full_legs'])
    copied_energy_legs = [l.tolist() for l in copied_energy_legs]
    copied_copied_legs = [l.tolist() for l in copied_environment_legs]
    k = [m for n in copied_copied_legs for m in n]
    out = np.arange(0, np.abs(np.min(k)))[::-1]

    for legs in copied_environment_legs:
        legs += np.abs(np.min(k))

    if len(dict_of_networks['bond']) == 2:
        for m,n in zip(dict_of_networks['environment'],
                       copied_environment_legs[:-2]):
            new_path.append(m.current_tensor)
            new_path.append(n)
        new_path.append(np.eye(2))
        new_path.append(copied_environment_legs[-2])
        new_path.append(np.eye(2))
        new_path.append(copied_environment_legs[-1])

    elif len(dict_of_networks['bond']) == 1:
        for m,n in zip(dict_of_networks['environment'],
                       copied_environment_legs[:-1]):
            new_path.append(m.current_tensor)
            new_path.append(n)
        new_path.append(np.eye(2))
        new_path.append(copied_environment_legs[-1])

    new_opt_path = oe.contract_path(*new_path, out, optimize=optimize_type)
    # print(copied_energy_legs)

    if len(dict_of_networks['bond']) == 2:
        for m,n in zip(dict_of_networks['entire_network'], copied_energy_legs[:-2]):
            new_path_energy.append(m.current_tensor)
            new_path_energy.append(n)
        # print(copied_energy_legs[:-2])
        # print([k.value for k in dict_of_networks['entire_network']])
        new_path_energy.append(np.eye(2))
        new_path_energy.append(copied_energy_legs[-2])
        new_path_energy.append(np.eye(2))
        new_path_energy.append(copied_energy_legs[-1])
    elif len(dict_of_networks['bond']) == 1:
        for m,n in zip(dict_of_networks['entire_network'], copied_energy_legs[:-1]):
            new_path_energy.append(m.current_tensor)
            new_path_energy.append(n)

        new_path_energy.append(np.eye(2))
        new_path_energy.append(copied_energy_legs[-1])

    new_opt_path_energy = oe.contract_path(*new_path_energy,optimize=optimize_type)
    # add new keys to existing dictionary
    dict_of_networks['einsum_path'] = new_opt_path[0]
    dict_of_networks['einsum_indices'] = copied_environment_legs
    dict_of_networks['out_list'] = out
    dict_of_networks['tensor_list'] = list_of_tensors
    dict_of_networks['einsum_energy_indices'] = copied_energy_legs
    dict_of_networks['einsum_path_energy'] = new_opt_path_energy[0]

# add environment or energy kwarg
def contract_network(operators, network, contract_type='env'):
    temp_operators = [i for i in operators[0]]
    path = []
    if contract_type == 'env':
        for m,n in zip(network['environment'], network['einsum_indices']):
            path.append(m.current_tensor)
            path.append(n)
        for m,n in zip(temp_operators, network['einsum_indices'][-len(temp_operators):]):
            path.append(m)
            path.append(n)

        return oe.contract(*path, network['out_list'],
                    optimize=network['einsum_path'])
    elif contract_type == 'energy':
        for m,n in zip(network['entire_network'], network['einsum_energy_indices']):
            path.append(m.current_tensor)
            path.append(n)

        for m,n in zip(temp_operators, network['einsum_energy_indices'][-len(temp_operators):]):
            path.append(m)
            path.append(n)
            # to_contract = [*[m.cur_tensor, n] in zip(network['entire_network'], network['full_legs'])]
        return oe.contract(*path, optimize=network['einsum_path_energy'])


def get_energy(tree_object, node):
    """ Docstring for get_energy() """
    temp = 0
    if tree_object.backend == 'torch':
        for operators in tree_object.hamiltonian:

            if operators[1] > 0:
                for network in node.vertical_networks:
                    # print('ver ',contract_network(operators, network,
                    # contract_type='energy').size())
                    temp += (contract_network(operators, network,
                            contract_type='energy')*operators[-1][0]).item()

                for network in node.horizontal_networks:
                    # print('hor ',contract_network(operators, network,
                    # contract_type='energy').size())
                    temp += (contract_network(operators, network,
                            contract_type='energy')*operators[-1][1]).item()
            else:
                for network in node.one_site_networks:
                    # print('one', contract_network(operators, network,
                    # contract_type='energy').size())
                    temp += (contract_network(operators, network,
                            contract_type='energy')*operators[-1][0]).item()

    elif tree_object.backend == 'numpy':
        for operators in tree_object.hamiltonian:

            if operators[1] > 0:
                for network in node.vertical_networks:
                    # print('ver ',contract_network(operators, network,
                    # contract_type='energy').size())
                    temp += contract_network(operators, network,
                            contract_type='energy')*operators[-1][0]

                for network in node.horizontal_networks:
                    # print('hor ',contract_network(operators, network,
                    # contract_type='energy').size())
                    temp += contract_network(operators, network,
                            contract_type='energy')*operators[-1][1]
            else:
                for network in node.one_site_networks:
                    # print('one', contract_network(operators, network,
                    # contract_type='energy').size())
                    temp += contract_network(operators, network,
                            contract_type='energy')*operators[-1][0]

    return temp

# @timer
def optimize_tensor(tree_object, node):
    """ VOID: optimize method for a single tensor in the Tree Tensor Network using optimal einsum"""

    if tree_object.backend == 'torch':
        node.cache_tensor.zero_()
        for operators in tree_object.hamiltonian:

            if operators[1] > 0:
                for network in node.vertical_networks:
                    node.cache_tensor.add_(contract_network(operators, network)*operators[-1][0])

                for network in node.horizontal_networks:
                    node.cache_tensor.add_(contract_network(operators, network)*operators[-1][1])
            else:
                for network in node.one_site_networks:
                    node.cache_tensor.add_(contract_network(operators, network)*operators[-1][0])

        new_shapes = node.cache_tensor.shape
        # need to transpose since torch saves n x n matrix of u, n being the first axes
        ut, s, v = node.cache_tensor.reshape(new_shapes[0], np.prod(new_shapes[1:])).T.svd(some=True)
        # torch returns u transposed hence ut.T
        node.current_tensor = -1.*torch.matmul(v,ut.T).reshape(new_shapes)

    elif tree_object.backend == 'numpy':

        node.cache_tensor.fill(0)
        for operators in tree_object.hamiltonian:

            if operators[1] > 0:
                for network in node.vertical_networks:
                    node.cache_tensor+=contract_network(operators, network)*operators[-1][0]

                for network in node.horizontal_networks:
                    node.cache_tensor+=contract_network(operators, network)*operators[-1][1]
            else:
                for network in node.one_site_networks:
                    node.cache_tensor+=contract_network(operators, network)*operators[-1][0]

        new_shapes = node.cache_tensor.shape
        u, s, v = np.linalg.svd(node.cache_tensor.reshape(new_shapes[0],
            np.prod(new_shapes[1:])).T, full_matrices = False)

        node.current_tensor =-1.*np.dot(v.T, u.T).reshape(new_shapes)

    for a_node in tree_object.node_list:
        if a_node.value == node.value:
            a_node.current_tensor = node.current_tensor

def exact_energy(N, hamiltonian, dimension):
    h = 0.0000
    if dimension == '2D' or dimension == '2d' or dimension == 2:
        print('computing 2D hamiltonian')
        L = int(N**0.5)
        for i in hamiltonian:
            if len(i[0]) == 1: # if length of operator list is 1
                operator = spl.csr_matrix(i[0][0])
                for k in range(N):
                    h += spl.kron(spl.kron(spl.identity(2**k), operator, 'csr'),
                        spl.identity(2**(N-k-1)), 'csr')*i[-1][0]
            if len(i[0]) == 2: # if length of operator list is 2
                if type(i[1]) == int:
                    operators = [spl.csr_matrix(j) for j in i[0]]

                    spacing_identity = spl.identity(2**(i[1]-1))
                    operators_2_site = spl.kron(spl.kron(operators[0], spacing_identity, 'csr'),
                        operators[1], 'csr')
                    # J+J- on i,i+1 and J-J+ on i-1,i
                    operators_2_site_reversed = spl.kron(spl.kron(operators[1], spacing_identity, 'csr'),
                        operators[0], 'csr')
                    # horizontal terms
                    for k in range(N):
                        if k%L:
                            h += spl.kron(spl.kron(spl.identity(2**(k-1)),operators_2_site, 'csr'),
                                spl.identity(2**(N-k-i[1])), 'csr')*i[2][0]
                    # vertical terms
                    for k in range(N-L):
                        h += spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**k),operators[0], 'csr'),
                            spl.identity(2**(L-1)), 'csr'), operators[1], 'csr'), spl.identity(2**(N-L-k-1)),'csr')*i[2][1]


                    # boundary terms
                    for k in range(L):
                        h += i[2][1]*spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**k),
                            operators[0], 'csr'),spl.identity(2**(N-L-1)), 'csr'),
                            operators[1], 'csr'), spl.identity(2**(L-k-1)), 'csr')

                        h += i[2][0]*(spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**(k*L)),
                            operators[0], 'csr'), spl.identity(2**(L-2)), 'csr'), operators[1], 'csr'),
                            spl.identity(2**(L*(L-k-1))), 'csr'))

                if type(i[1]) == float:
                    if i[1] == 1.5:
                        # upwards -> *i[2][1], downwards *i[2][0]
                        operators = [spl.csr_matrix(j) for j in i[0]]
                        for k in range(N-L):
                            if k%L:
                                # downward to the right
                                left_id, mid_id, right_id = k-1, L, N-L-(k-1)-2
                                # print(left_id, mid_id, right_id, left_id+mid_id+right_id)
                                h+=spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**left_id), operators[0], 'csr'), spl.identity(2**mid_id), 'csr'),
                                    operators[1], 'csr'), spl.identity(2**right_id),'csr')*i[2][0]

                                # upward to the right
                                left_id_up, mid_id_up, right_id_up = k, L-2, N-(L-2)-k-2
                                # print(left_id_up, 1, mid_id_up, 1, right_id_up)
                                h+=spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**left_id_up), operators[1], 'csr'), spl.identity(2**mid_id_up), 'csr'),
                                    operators[0], 'csr'), spl.identity(2**right_id_up),'csr')*i[2][1]
                        # boundaries right up, followed by right down:
                        for k in range(L-1):
                            # up
                            left_id, mid_id = L*k, 2*L-2
                            right_id = N - left_id - mid_id-2
                            # print(left_id, mid_id, right_id, left_id+mid_id+right_id)
                            h+=spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**left_id), operators[1], 'csr'), spl.identity(2**mid_id), 'csr'),
                                operators[0], 'csr'), spl.identity(2**right_id),'csr')*i[2][1]

                            # down
                            left_id_down, mid_id_down = L*(k+1)-1, 0
                            right_id_down = N-left_id_down-mid_id_down - 2
                            # print(left_id_down, mid_id_down, right_id_down)
                            h+=spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**left_id_down), operators[0], 'csr'), spl.identity(2**mid_id_down), 'csr'),
                                operators[1], 'csr'), spl.identity(2**right_id_down),'csr')*i[2][0]

                        # boundaries, followed by up right:
                        # for up left the first boundary is omitted and calculated under
                        # boundary top right upwards (to bottem):

                        for k in range(L-1):
                            # boundary top right upwards (to bottem):
                            left_id = k
                            mid_id = L*(L-1)
                            right_id = N - 2 - mid_id - left_id
                            # print(left_id, mid_id, right_id)
                            h+=spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**left_id), operators[0], 'csr'), spl.identity(2**mid_id), 'csr'),
                                operators[1], 'csr'), spl.identity(2**right_id),'csr')*i[2][1]

                            left_id2 = k+1
                            mid_id2 = L*(L-1) - 2
                            right_id2 = N - 2 -left_id2 - mid_id2
                            # print(left_id2, mid_id2, right_id2)
                            # boundary bottem right downwards to top
                            # order of operators is correct (trust me)
                            h+=spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**left_id2), operators[1], 'csr'), spl.identity(2**mid_id2), 'csr'),
                                operators[0], 'csr'), spl.identity(2**right_id2),'csr')*i[2][0]

                        # boundary top right [op1] to bottem left [op2] so upwards to the right:
                        h += spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**(L-1)), operators[0], 'csr'), spl.identity(2**((L-2)*L)), 'csr'),
                            operators[1], 'csr'), spl.identity(2**(L-1)),'csr')*i[2][1]
                        # boundary bottem right [op1] to top left [op2] so downwards to the right:
                        h += spl.kron(spl.kron(spl.kron(spl.kron(spl.identity(2**(0)), operators[1], 'csr'), spl.identity(2**(N-2)), 'csr'),
                            operators[0], 'csr'), spl.identity(2**(0)),'csr')*i[2][0]


        return h

def rho_bot_sites(tree_object, sites):
        temporary_network = []
        for tensor in tree_object.node_list:
            for site in sites:
                if site in tensor.lattice:
                    temporary_network.append(tensor)

        unique_network = []
        for i in temporary_network:
            if i not in unique_network:
                unique_network.append(i)
            else:
                continue

        new_open_legs = np.arange(1, 2*(len(sites))+1)*-1
        new_bra_open_legs, new_ket_open_legs = np.array_split(new_open_legs, 2)
        new_bra_open_legs, new_ket_open_legs = list(reversed(new_bra_open_legs.tolist())), list(reversed(new_ket_open_legs.tolist()))
        all_legs = []

        for current_node in unique_network:
            if current_node.isRoot():

                current_node.bralegs = np.array([1,2,3])
                current_node.ketlegs = np.array([1, None, None])

                if current_node.left in unique_network and not current_node.right in unique_network:
                    current_node.ketlegs[2] = current_node.bralegs[2]
                elif current_node.right in unique_network and not current_node.left in unique_network:
                    current_node.ketlegs[1] = current_node.bralegs[1]

                mask_legs = np.where(current_node.ketlegs == None)[0]
                new_values = np.arange(np.max(current_node.bralegs)+1,
                    np.max(current_node.bralegs)+mask_legs.size+1)
                current_node.ketlegs[mask_legs] = new_values
                max_leg = np.max(np.array([np.max(current_node.ketlegs), np.max(current_node.bralegs)]))

            if not current_node.isRoot():
                current_node.bralegs = [None]*len(current_node.current_tensor.shape)
                current_node.ketlegs = [None]*len(current_node.current_tensor.shape)
                current_node.bralegs, current_node.ketlegs = np.array(current_node.bralegs), np.array(current_node.ketlegs)

                if current_node.isLeftChild():
                    current_node.bralegs[0] = current_node.parent.bralegs[1]
                    current_node.ketlegs[0] = current_node.parent.ketlegs[1]

                if current_node.isRightChild():
                    current_node.bralegs[0] = current_node.parent.bralegs[2]
                    current_node.ketlegs[0] = current_node.parent.ketlegs[2]

                if current_node.layer != tree_object.cut:
                    mask_legs = np.where(current_node.bralegs == None)[0]
                    new_bralegs = np.arange(max_leg+1, max_leg+mask_legs.size+1)
                    current_node.bralegs[mask_legs] = new_bralegs
                    max_leg = np.max(current_node.bralegs)

                # for lower legs
                if current_node.left in unique_network and current_node.right in unique_network:
                    mask_legs = np.where(current_node.ketlegs == None)[0]
                    current_node.ketlegs[mask_legs] = np.arange(max_leg+1, max_leg+mask_legs.size+1)
                    max_leg = np.max(current_node.ketlegs)

                elif current_node.left in unique_network and not current_node.right in unique_network:
                    current_node.ketlegs[2] = current_node.bralegs[2]
                    current_node.ketlegs[1] = max_leg+1
                    max_leg = np.max(current_node.ketlegs)

                elif current_node.right in unique_network and not current_node.left in unique_network:
                    current_node.ketlegs[1] = current_node.bralegs[1]
                    current_node.ketlegs[2] = max_leg+1
                    max_leg = np.max(current_node.ketlegs)

                # if current_node = bottem tensor
                elif not current_node.left in unique_network and not current_node.right in unique_network:
                    for site in sites:
                        if site in current_node.lattice.flatten():
                            index_to_mask = np.where(current_node.lattice.flatten() == site)[0]+1
                            current_node.bralegs[index_to_mask] = new_bra_open_legs.pop()
                            current_node.ketlegs[index_to_mask] = new_ket_open_legs.pop()
                    legs_to_mask = np.where(current_node.bralegs == None)[0]
                    new_closed_legs = np.arange(max_leg+1, legs_to_mask.size+max_leg+1)
                    current_node.bralegs[legs_to_mask], current_node.ketlegs[legs_to_mask] = new_closed_legs, new_closed_legs
                    max_leg = np.max(np.array([current_node.ketlegs, np.array(current_node.bralegs)]))
            all_legs.extend((current_node.bralegs, current_node.ketlegs))
        reduced_density_matrix_list = []
        order_legs = [i for i in range(1, max_leg+1)][::-1]
        for i in unique_network:
            reduced_density_matrix_list.extend((i,i))
        new_shape = 2**int(new_open_legs.size/2)

        # print(all_legs)
        all_legs_2 = [np.copy(l) for l in all_legs]
        tensor_list = []
        f = [l.tolist() for l in all_legs_2]
        k = [m for n in f for m in n]

        out = np.arange(0, np.abs(np.min(k)))[::-1]
        for s2 in all_legs_2:
            s2+= np.abs(np.min(k))
        # print(all_legs_2)
        new_path = []
        # print(out)
        for m,n in zip(reduced_density_matrix_list, all_legs_2):
            new_path.append(m.current_tensor)
            new_path.append(n)

        og_reduced_density_matrix = oe.contract(*new_path, out, optimize='greedy')
        reduced_density_matrix = og_reduced_density_matrix.reshape(new_shape, new_shape)
        return og_reduced_density_matrix, reduced_density_matrix


def rho_layer(tree_object, layer):
    # TODO: write this function (for any n-layer binary tree)
    """ Docstring for rho_layer() """
    temp_rho_list = []
    for a_node in tree_object.node_list:
        if (a_node.layer <= layer) and (a_node.value[-1] != '1'):
            temp_rho_list.append(a_node)
    temp_rho_list.sort(key = lambda x: x.layer)
    if tree_object.backend == 'torch':
        legs = np.arange(1, len(tree_object.root.current_tensor.size())*(layer+1)+2)
    elif tree_object.backend == 'numpy':
        legs = np.arange(1, len(tree_object.root.current_tensor.shape)*(layer+1)+2)

    print(legs)

    print([j.value for j in temp_rho_list])


def two_point_correlator(tree_object, operators):
    # TODO: write this function (can use rho_bot_sites)
    pass


def mean_two_point_correlatior_i_ir(tree_object, operators):
    # TODO: write this function using two_point_correlator
    pass
