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
        self.bondlist = None
        self.bclist = None
        self.tensornet = []
        self.bralegs = [None]*3
        self.ketlegs =[None]*3
        self.tensor_dicts = []

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

    def __init__(self, root, cut = None, chilist = None, hamiltonian = None,
        dimension = None, bc_type = 'closed', tree_seed = None,
        forbidden_bonds = [], optimize_type = 'greedy', backend='torch'):

        self.chilist=chilist
        self.backend = backend
        self.optimize_type = optimize_type
        self.root = root
        self.dimension = dimension
        self.hamiltonian = hamiltonian
        self.cut = cut
        self.tree_seed = tree_seed
        self.node_list = []
        self.times = []
        self.energy_per_sweep_list = []
        self.current_expectation_values = None
        self.current_iteration = 0
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
        self.get_bonds(self.root.left)


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

    def insert_tensor(self, node):
        """ Inserts tensors in the node attribute 'current_tensor' given bond dimension list
            named 'chilist.'"""

        if node is self.root:
            node.current_tensor = tt.create_sym_tensor(1,
                self.chilist[node.layer], self.chilist[node.layer], self.backend)

        elif node.layer == self.cut:
            if node.lattice.size%2 == 0:
                node.current_tensor = tt.create_sym_tensor(self.chilist[-1],
                    int(2**(len(node.lattice)/2)), int(2**(len(node.lattice)/2))).reshape(self.chilist[-1],
                        *np.ones(len(node.lattice), dtype = 'int')*2, self.backend)
            else:
                # print('not mod 2')
                node.current_tensor = (tt.create_tensor(self.chilist[-1],
                                    int(2**(node.lattice.size))).
                                    reshape(self.chilist[-1],*np.ones(node.lattice
                                    .size, dtype = 'int')*2))
        else:
            node.current_tensor = tt.create_sym_tensor(self.chilist[node.parent.layer],
                self.chilist[node.layer], self.chilist[node.layer], self.backend)

    def get_bonds(self, node):
        """ finds the bonds of the current node according to a given hamiltonian
            and dimension """
        print(node.lattice, node.lattice.shape,'\n',
            self.root.lattice.shape, self.root.lattice)
        for i in node.lattice.flatten():
            locations = np.where(self.root.lattice==i)
            print(locations, type(locations))
            # m, n = locations[0][0], locations[0][1]
            # vertical_bond = [self.root.lattice[m,n], self.root.lattice[(m+1)%self.square_size,n]]
            # print(vertical_bond)
        # print(np.where(node.lattice))

        for operator in self.hamiltonian:
            pass
