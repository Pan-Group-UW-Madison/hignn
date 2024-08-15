from Edge_construction_transfer import create_edge_3body
import torch
from torch import Tensor
import numpy as np
import time
from Morse_force import cal_Morse_force

# x = np.array([0.0, 0.0, 0.0, -3.7, 9.9, -8.4, 0.7, -1.2, -5.3])
x = np.loadtxt('data_sample.txt_test', dtype="f8")
Nc = x.shape[0]
n_dim = 3
# force = Tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]).reshape((3, 3))
# force = torch.zeros((Nc, 3))
# force[:, 0] = torch.arange(0, 1000, 1)
# force = cal_Morse_force(x, Nc, 5, -16, 16, 5)
# force = torch.from_numpy(force)
# print(force.shape)
# print(force[0:10, :])
# R_cut = 4

t1 = time.time()
edge_index1, edge_attr1, edge_indexs, edge_attrs = create_edge_3body(
    x, Nc, n_dim, R_cut, force)
t2 = time.time()
print(t2 - t1)

print(edge_index1.shape)
print(edge_attr1.shape)
print(edge_indexs.shape)
print(edge_attrs.shape)
