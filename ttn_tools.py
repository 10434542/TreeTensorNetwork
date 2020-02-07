import numpy as np
import torch
import time
import opt_einsum as oe
import cupy as cp
################################################################################
# TOOLS FOR TTN file                                                           #
################################################################################
torch.set_printoptions(10)

def create_cache_tensor(*dims, backend='torch'):
    for i in dims:
        if type(i) == float:
            print(i, type(i))
            print("type is not int m8")
            raise TypeError
    if backend == 'torch':
        tens = torch.zeros(*dims, dtype = torch.double, device='cuda:0')
    elif backend == 'numpy':
        tens = np.zeros(dims)
    return tens

def create_tensor(*dims, backend = 'torch'):
    for i in dims:
        if type(i) == float:
            print(i, type(i))
            print("type is not int m8")
            raise TypeError
    if backend == 'torch':
        tens = torch.rand(*dims, dtype=torch.double, device='cuda:0')
        tens = tens.T.svd(some=True)[0]
        tens = tens.T.reshape(*dims)
    elif backend == 'numpy':
        tens = np.random.rand(*dims)
        tens = np.linalg.svd(tens, full_matrices = False)[-1]
    return tens

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
        # tens = torch.rand(*dims,dtype = torch.double, device='cuda:0')
        tens = torch.cuda.DoubleTensor(*dims).random_(0, 1)
        tens = tens+tens.transpose(2,1)
        # transpose is need, for explanation see:
        # https://github.com/pytorch/pytorch/issues/24900
        tens = tens.reshape(dims[0],dims[1]*dims[1]).T
        u, s, v = tens.svd(some=True)
        tens = u.T.reshape(*dims)
        return tens

    elif backend=='numpy':
        tens = np.random.uniform(0,10,[*dims])
        tens = tens + np.transpose(tens, (0, 2, 1))
        tens = np.linalg.svd(tens.reshape(dims[0],dims[1]**2), full_matrices=False)[-1]
        tens = tens.reshape(*dims)

    else:
        tens=None

    return tens

def get_absolute_distance(lattice, spacing, constraints):
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
                        mask_site = np.where(current_node.lattice == j)[0]+1

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

def contract_network(operators, network):
    temp_operators = [i for i in operators[0]]
    path = []
    # begin = time.time()
    for m,n in zip(network['environment'], network['einsum_indices']):
        path.append(m.current_tensor)
        path.append(n)
    for m,n in zip(temp_operators, network['einsum_indices'][-len(temp_operators):]):
        path.append(m)
        path.append(n)
    # end = time.time()
    # print('loop time per bond: ', end-begin)
    temp_tens = operators[-1][0]*oe.contract(*path, network['out_list'],
                    optimize=network['einsum_path'])

    # print(temp_tens.shape)
    return temp_tens

def optimize_tensor(tree_object, node):
    """ VOID: optimize method for a single tensor in the Tree Tensor Network using optimal einsum"""
    print(node.cache_tensor)
    ti1 = time.time()
    for operators in tree_object.hamiltonian:
        print(operators[1])
        if operators[1] > 0:
            for network in node.vertical_networks:
                node.cache_tensor.add_(contract_network(operators, network))

            for network in node.horizontal_networks:
                node.cache_tensor.add_(contract_network(operators, network))

        else:
            for network in node.one_site_networks:
                node.cache_tensor.add_(contract_network(operators, network))
    tf1 = time.time()
    if tree_object.backend == 'torch':
        torch.cuda.synchronize()
        print("contract time took %s sec"%(tf1 - ti1))
        print(node.cache_tensor)
        ti = time.time()
        new_shapes = node.cache_tensor.shape
        ut, s, v = node.cache_tensor.reshape(new_shapes[0], np.prod(new_shapes[1:])).T.svd(some=True)
        torch.cuda.synchronize()
        tf = time.time()
        node.current_tensor = -1.*torch.matmul(ut,v).reshape(new_shapes)
        node.cache_tensor.zero_()
        print("svd time torch took %s sec"%(tf-ti))

    elif tree_object.backend == 'numpy':
        tf1 = time.time()
        print("contract time took %s sec"%(tf - ti))
        ti = time.time()
        new_shapes = node.cache_tensor.shape
        u, s, v = np.linalg.svd(node.cache_tensor.reshape(new_shapes[0],
            np.prod(new_shapes[1:])).T, full_matrices = False)
        tf = time.time()
        print("svd time numpy took %s sec"%(tf-ti))
        node.current_tensor =-1.*np.dot(u,v).reshape(new_shapes)
        node.cache_tensor.fill(0)
