import ttn_tools as tt
import numpy as np

class Node:
    """Base Node class for TreeTensorNetwork.
       ------------------------------------------------------------------------

    """
    def __init__(self, value, lattice = None, parent = None, right = None,
        left = None, layer = None):
        self.value = value
        self.parent = parent
        self.right = right
        self.left = left
        self.lattice = lattice
        self.layer = layer
        self.current_tensor = None
        self.cache_tensor = None
        self.vertical_two_site_terms = None
        self.horizontal_two_site_terms = None
        self.one_site_terms = None
        self.vertical_bc_terms = None
        self.horizontal_bc_terms = None
        self.t_shape = None
        self.horizontal_networks = []
        self.vertical_networks = []
        self.one_site_networks = []
        self.bralegs = [None]*3
        self.ketlegs =[None]*3

    def __str__(self):
        return (str(self.layer))

    def hasRightChild(self):
        return self.right

    def hasLeftChild(self):
        return self.left

    def isLeftChild(self):
        return self.parent and self.parent.left == self

    def isRightChild(self):
        return self.parent and self.parent.right == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.right or self.left)

    def hasAnyChildren(self):
        return self.right or self.left

    def hasBothChildren(self):
        return self.right and self.left

class TreeTensorNetwork:
    """docstring for TreeTensorNetwork
        -----------------------------------------------------------------------

    """

    def __init__(self, system_size, cut = None, chilist = None, hamiltonian = None,
        dimension = None, bc_type = 'closed', tree_seed = None,
        forbidden_bonds = [], optimize_type = 'greedy', backend='torch'):

        self.root = Node(value='0', layer=0, lattice=np.arange(1,system_size+1))
        self.chilist=chilist
        self.backend = backend
        self.optimize_type = optimize_type
        self.dimension = dimension
        self.hamiltonian = hamiltonian
        self.cut = cut
        self.tree_seed = tree_seed
        self.node_list = []
        self.times = []
        self.energy_per_sweep_list = []
        self.current_expectation_values = None
        self.current_iteration = 0
        # # TODO: add hamiltonian functionality
        self.spacings = np.unique([i[1] for i in self.hamiltonian])
        self.bc_type = str.lower(bc_type)
        self.square_size = np.sqrt(self.root.lattice.size).astype(int)
        self.reference_lattice = np.copy(self.root.lattice)
        self.root.lattice = self.root.lattice.reshape(self.square_size,self.square_size)
        # assert(len(self.chilist)==cut), 'the length of list of bond-dimensions must equal the cut'
        #
        # self.insert_nodes(self.root)
        # self.flatten_lattices()
        # self.forbidden_bonds = self.get_boundary_bonds(self.bc_type)
        # self.set_attributes()
        self.insert_nodes(self.root)
        self.set_bonds()
        for i in self.node_list:
            self.insert_tensor_v2(i)
        self.prepare_networks()
        self.add_legs()
        self.get_orders()


    def store_time(self, t):
        """ method that merely serves to store time using decorator timer()"""
        self.times.append(t)


    def insert_nodes(self, node):
        """ docstring for insert_nodes """
        self.node_list.append(node)
        # print(node.layer, node.value, node.layer+1)
        # print('reference_lattice shape:', self.reference_lattice.shape)
        if node.left:
            node = node.left
        elif (not node.left) and (node.layer < self.cut) and len(node.lattice.flatten())%2==0:
            # print(node.layer+1, node.lattice.shape)
            # print('hallo')
            if (node.layer +1)%2 ==0:
                # print('hey')
                temp_lattice = np.hsplit(node.lattice,2)[0]
            elif (node.layer +1)%2 != 0:
                # print('hoi')
                temp_lattice = np.vsplit(node.lattice,2)[0]

            node.left = Node(value = node.value+'0',parent = node,
                  layer = node.layer+1, lattice = temp_lattice)
            self.insert_nodes(node.left)
        if node.right:
            node = node.right
        elif not node.right and (node.layer < self.cut) and len(node.lattice.flatten())%2==0:
            # node.right = Node(value = node.value +'1', parent = node,
            #     layer = node.layer+1, lattice = node.lattice[int(len(node.lattice)/2):])
            if (node.layer +1)%2 ==0:
                temp_lattice = np.hsplit(node.lattice,2)[1]
            elif (node.layer +1)%2 != 0:
                temp_lattice = np.vsplit(node.lattice,2)[1]
            node.right = Node(value = node.value+'1',parent = node,
                  layer = node.layer+1, lattice = temp_lattice)
            self.insert_nodes(node.right)

    def insert_tensor_v2(self, node):
        if node is self.root:
            node.current_tensor = tt.create_sym_tensor(1,
                self.chilist[node.layer], self.chilist[node.layer], backend=self.backend)
            node.cache_tensor = tt.create_cache_tensor(1,
                self.chilist[node.layer], self.chilist[node.layer], backend=self.backend)

        elif (node.layer == self.cut) and (node.isLeftChild):
            if node.lattice.flatten().size%2 == 0:
                node.current_tensor = tt.create_sym_tensor(self.chilist[-1],
                    int(2**(len(node.lattice.flatten())/2)), int(2**(len(node.lattice.flatten())/2)),
                    backend=self.backend).reshape(self.chilist[-1],
                        *np.ones(len(node.lattice.flatten()), dtype = 'int')*2)
                node.cache_tensor= tt.create_cache_tensor(self.chilist[-1],
                    int(2**(len(node.lattice.flatten())/2)), int(2**(len(node.lattice.flatten())/2)),
                    backend=self.backend).reshape(self.chilist[-1],
                        *np.ones(len(node.lattice.flatten()), dtype = 'int')*2)
            else:
                # print('not mod 2')
                node.current_tensor = (tt.create_tensor(self.chilist[-1],
                                    int(2**(node.lattice.size)), backend=self.backend).
                                    reshape(self.chilist[-1],*np.ones(node.lattice
                                    .size, dtype = 'int')*2))
                node.cache_tensor = (tt.create_cache_tensor(self.chilist[-1],
                                    int(2**(node.lattice.size)), backend=self.backend).
                                    reshape(self.chilist[-1],*np.ones(node.lattice
                                    .size, dtype = 'int')*2))

        elif (node.layer != self.cut) and (node.isLeftChild) and (node is not self.root):
            node.current_tensor = tt.create_sym_tensor(self.chilist[node.parent.layer],
                self.chilist[node.layer], self.chilist[node.layer], backend=self.backend)
            node.cache_tensor = tt.create_cache_tensor(self.chilist[node.parent.layer],
                self.chilist[node.layer], self.chilist[node.layer], backend=self.backend)

        # all nodes within a layer have the same tensor!
        for another_node in self.node_list:
            if (another_node.layer == node.layer) and (another_node.value != node.value):
                another_node.current_tensor = node.current_tensor
                another_node.cache_tensor = node.cache_tensor

    def insert_tensor(self, node):
        """ Inserts tensors in the node attribute 'current_tensor' given bond dimension list
            named 'chilist.'"""

        if node is self.root:
            node.current_tensor = tt.create_sym_tensor(1,
                self.chilist[node.layer], self.chilist[node.layer], backend=self.backend)
            node.cache_tensor = tt.create_cache_tensor(1,
                self.chilist[node.layer], self.chilist[node.layer], backend=self.backend)

        elif node.layer == self.cut:
            if node.lattice.flatten().size%2 == 0:
                node.current_tensor = tt.create_sym_tensor(self.chilist[-1],
                    int(2**(len(node.lattice.flatten())/2)), int(2**(len(node.lattice.flatten())/2)),
                    backend=self.backend).reshape(self.chilist[-1],
                        *np.ones(len(node.lattice.flatten()), dtype = 'int')*2)
                node.cache_tensor= tt.create_cache_tensor(self.chilist[-1],
                    int(2**(len(node.lattice.flatten())/2)), int(2**(len(node.lattice.flatten())/2)),
                    backend=self.backend).reshape(self.chilist[-1],
                        *np.ones(len(node.lattice.flatten()), dtype = 'int')*2)

            # TODO: make a function create_tensor and rewrite code below
            else:
                # print('not mod 2')
                node.current_tensor = (tt.create_tensor(self.chilist[-1],
                                    int(2**(node.lattice.size))).
                                    reshape(self.chilist[-1],*np.ones(node.lattice
                                    .size, dtype = 'int')*2))
                node.cache_tensor = (tt.create_cache_tensor(self.chilist[-1],
                                    int(2**(node.lattice.size))).
                                    reshape(self.chilist[-1],*np.ones(node.lattice
                                    .size, dtype = 'int')*2))
        else:
            node.current_tensor = tt.create_sym_tensor(self.chilist[node.parent.layer],
                self.chilist[node.layer], self.chilist[node.layer], backend=self.backend)
            node.cache_tensor = tt.create_cache_tensor(self.chilist[node.parent.layer],
                self.chilist[node.layer], self.chilist[node.layer], backend=self.backend)

    def set_bonds(self):
        """ Method to be placed in init of class """
        for i in self.node_list:
            temp_bonds = tt.get_bonds(self.root.lattice, i.lattice, self.spacings)
            (i.horizontal_two_site_terms, i.vertical_two_site_terms, i.one_site_terms,
            i.vertical_bc_terms, i.horizontal_bc_terms) = temp_bonds


    def get_orders(self):
        for node in self.node_list:
            for network in node.vertical_networks:
                tt.get_optimal_order(node, network, self.optimize_type)
            for network in node.horizontal_networks:
                tt.get_optimal_order(node, network, self.optimize_type)
            for network in node.one_site_networks:
                tt.get_optimal_order(node, network, self.optimize_type)


    def tensors_to(self, chip_type):

        if self.backend == 'torch':
            for node in self.node_list:
                print('loading Node %s onto %s'%(node.value, chip_type))
                node.current_tensor = node.current_tensor.to(device=chip_type)
                # node.cache_tensor.to("cuda")
                print('Node %s its tensor loaded onto %s'%(node.value, chip_type))
        else:
            print("try using torch as 'backend'")
            return


    def add_legs(self):
        """ Docstring for add_legs """
        for node in self.node_list:
            tt.get_legs(self.cut, node, node.vertical_networks)
            tt.get_legs(self.cut, node, node.horizontal_networks)
            tt.get_legs(self.cut, node, node.one_site_networks)

    def prepare_networks(self):
        """ Docstring for prepare_networks """
        # can't zip since lengths of lists can differ in size
        for node in self.node_list:
            for a_bond in node.vertical_two_site_terms:
                node.vertical_networks.append(tt.get_single_network(self.node_list, a_bond))
            for a_bond in node.horizontal_two_site_terms:
                node.horizontal_networks.append(tt.get_single_network(self.node_list, a_bond))
            for a_bond in node.vertical_bc_terms:
                node.vertical_networks.append(tt.get_single_network(self.node_list, a_bond))
            for a_bond in node.horizontal_bc_terms:
                node.horizontal_networks.append(tt.get_single_network(self.node_list, a_bond))
            for a_bond in node.one_site_terms:
                node.one_site_networks.append(tt.get_single_network(self.node_list, a_bond))
