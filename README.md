# TreeTensorNetwork
Code written for my master thesis titled "Tree Tensor Networks in 1D & 2D" (See thesis for an elaborate explanation!). Where applications to the Transverse Field Ising Model and the J1-J2 Heisenberg model using a Binary TTN have been studied in detail.

Before using this code I would advice you to take a look a the requirements.txt so you can install the proper libraries needed. In addition a dockerfile has been added which I have used during the last couple of months of the project. Simple type "run some_python_file.py" and I will get the job done after having build the docker.

So how does one use this code?
The core files of this module are "ttn.py", "ttn_tools.py", "torch_hamiltonians.py", and "numpy_hamiltonians.py" which you can import in a new file to get started.
We then continue by creating a "TreeTensorNetwork" object from the ttn.py file and parse the system size, amount of cuts of the lattice/chain, list of the bond-dimensions, the hamiltonian, the dimension, the boundary conditions, and a seed if needed. Furthermore we can also provide a backend which can be either "numpy" or "torch" which in turn determines from which file your hamiltonian must originate from. 
Once we have initialized our TreeTensorNetwork we can use the tools (functions) from the "ttn_tools.py" file and apply them on the TreeTensorNetwork (TTN) object.
So far these tools consists of optimization functions, correlators, write/read functions, and some other supplementary functions.

Some remarks:
I have implemented some spatial symmetries to save computational costs since I did not enjoy enough RAM.
Feel FREE to adjust and critique the code!

If there are any questions regarding the code please contact me for help.
