import torch
import opt_einsum as oe
from timeit import default_timer as timer
import numpy as np

A = torch.rand(200,200, dtype = torch.double, device = "cuda")
B = torch.rand(200,200,200, dtype = torch.double, device = "cuda")
start = timer()
for i in range(10):
    C = oe.contract("jk, jlm, hnm, jk -> ln", A,B,B, A, backend="torch")
end = timer()
print(end-start)
print(type(C))
A = np.random.random((200,200))
B = np.random.random((200,200, 200))

start = timer()
for i in range(10):
    C = oe.contract("jk, jlm, hnm, jk -> ln", A,B,B, A, backend="numpy")
end = timer()
print(end - start)
print(type(C))
