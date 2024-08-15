import torch
import numpy as np
from Data3 import Data3
from edge_conv3 import EdgeConv3
from Dataloader3 import DataLoader
import torch.nn as nn
from ase import Atoms
import random
import time
from scipy import spatial

# x, y = np.mgrid[0:4, 0:4]
# points = np.c_[x.ravel(), y.ravel()]
# print(points)
# tree = spatial.cKDTree(points)
#
# print(tree.query_ball_point([2, 2], 1))


raw_data = np.loadtxt('../data/Stokesian_10648cir_large.txt', dtype="f8")
print(raw_data.shape)
Nc = 6400
raw_data = raw_data[[0, 1], 0:6 * Nc]
print(raw_data.shape)

x_index = np.arange(0, 6 * Nc, 6)
y_index = np.arange(1, 6 * Nc, 6)
z_index = np.arange(2, 6 * Nc, 6)
vx_index = np.arange(3, 6 * Nc, 6)
vy_index = np.arange(4, 6 * Nc, 6)
vz_index = np.arange(5, 6 * Nc, 6)
raw_data[:, vz_index] = raw_data[:, vz_index] - 1.0
raw_data[:, vx_index] = raw_data[:, vx_index] * (-6 * np.pi)
raw_data[:, vy_index] = raw_data[:, vy_index] * (-6 * np.pi)

x_min = 0.0001918401
x_max = 65.999689
y_min = 0.0001363674
y_max = 65.99998
z_min = 0.00036502483
z_max = 65.999783

raw_data[:, x_index] = (raw_data[:, x_index] - x_min) / (x_max - x_min)
raw_data[:, y_index] = (raw_data[:, y_index] - y_min) / (y_max - y_min)
raw_data[:, z_index] = (raw_data[:, z_index] - z_min) / (z_max - z_min)

np.random.shuffle(raw_data)
scale = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
R_cutoff_2body = 1000
R_cutoff_3body = 8


def compute_real_dis(pos1, pos2, scale):
    x1 = (pos1[0] - pos2[0]) * scale[0]
    y1 = (pos1[1] - pos2[1]) * scale[1]
    z1 = (pos1[2] - pos2[2]) * scale[2]
    dis = np.sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2))
    return dis


def create_edge(row, Nc, scale, R_cutoff):
    edge_info = np.zeros((2, Nc * Nc))
    a1 = np.arange(0, Nc, 1)
    a1 = np.repeat(a1, axis=0, repeats=Nc)
    a2 = np.arange(0, Nc, 1).reshape(1, Nc)
    a2 = np.repeat(a2, axis=0, repeats=Nc)
    a2 = a2.reshape(1, Nc * Nc)
    edge_info[0, :] = a1
    edge_info[1, :] = a2
    return edge_info


def create_edge3(row, Nc, R_cutoff):
    R_cut = R_cutoff
    tree = spatial.cKDTree(row)
    edge_info = np.zeros((Nc * Nc * 10, 3))
    kk = 0
    for i in range(Nc):
        edge1 = tree.query_ball_point(row[i, :], R_cut)
        edge1.remove(i)
        edge1_array = np.array(edge1)
        tree2 = spatial.cKDTree(row[edge1, :])
        for j in range(len(edge1)):
            edge2 = tree2.query_ball_point(row[edge1[j], :], R_cut)
            edge2.remove(j)
            edge_info[kk:(kk + len(edge2)), 0] = edge1_array[edge2]
            edge_info[kk:(kk + len(edge2)), 1] = edge1[j]
            edge_info[kk:(kk + len(edge2)), 2] = i
            kk = kk + len(edge2)
    print(kk)
    edge_info = edge_info[0:kk, :].T
    return edge_info


def create_data(raw_data, train_index, val_index, scale, R_cutoff_2body, R_cutoff_3body):
    Nc = int(raw_data.shape[1] / 6)
    N_data = raw_data.shape[0]
    x_index = np.arange(0, 6 * Nc, 6)
    y_index = np.arange(1, 6 * Nc, 6)
    z_index = np.arange(2, 6 * Nc, 6)
    vx_index = np.arange(3, 6 * Nc, 6)
    vy_index = np.arange(4, 6 * Nc, 6)
    vz_index = np.arange(5, 6 * Nc, 6)
    idx_x = np.concatenate((x_index, y_index, z_index), axis=0)
    idx_x = np.sort(idx_x)
    # idx_y = vz_index
    idx_y = np.concatenate((vx_index, vy_index, vz_index), axis=0)
    idx_y = np.sort(idx_y)
    data_list = list()
    for i in range(N_data):
        # if i % 1 == 0:
        #     print(i)
        # edge_index = create_edge(raw_data[i, :], Nc, scale, R_cutoff_2body)
        edge_index = np.zeros((2, 0))
        edge_index = torch.from_numpy(edge_index)
        edge_index = edge_index.long()
        edge_index1 = create_edge3(raw_data[i, idx_x].reshape(Nc, 3), Nc, R_cutoff_3body/115)
        # edge_index1 = np.zeros((3, 0))
        edge_index1 = torch.from_numpy(edge_index1)
        edge_index1 = edge_index1.long()
        x = torch.tensor(raw_data[i, idx_x].reshape((Nc, 3)), dtype=torch.float)
        y = torch.tensor(raw_data[i, idx_y].reshape((Nc, 3)), dtype=torch.float)
        data = Data3(x=x, y=y, edge_index=edge_index, edge_index1=edge_index1)
        data_list.append(data)

    return data_list


t1 = time.time()
data_list1 = create_data(raw_data, 25000, 30000, scale, R_cutoff_2body, R_cutoff_3body)
data = data_list1[0]
# print(data.edge_index1)
t2 = time.time()
time1 = (t2 - t1) / raw_data.shape[0]
print(f'Construction data time: {time1:.9f}')

# is_gpu = torch.cuda.is_available()
# torch.cuda.set_device(0)
#
#
# # is_gpu = False
#
#
# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels11, hidden_channels12, hidden_channels13, hidden_channels21, hidden_channels22,
#                  hidden_channels23):
#         super(GCN, self).__init__()
#         self.net1 = nn.Sequential(
#             nn.Linear(6, hidden_channels11),
#             nn.ReLU(),
#             nn.Linear(hidden_channels11, hidden_channels12),
#             nn.ReLU(),
#             nn.Linear(hidden_channels12, hidden_channels13),
#             nn.ReLU(),
#             nn.Linear(hidden_channels13, 3)
#         )
#         self.net2 = nn.Sequential(
#             nn.Linear(9, hidden_channels21),
#             nn.ReLU(),
#             nn.Linear(hidden_channels21, hidden_channels22),
#             nn.ReLU(),
#             nn.Linear(hidden_channels22, hidden_channels23),
#             nn.ReLU(),
#             nn.Linear(hidden_channels23, 3)
#         )
#         # torch.manual_seed(123)
#         self.edgeconv1 = EdgeConv3(nn_2body=self.net1, nn_3body=self.net2, aggr='add')
#         # self.conv1 = GraphConv(3, hidden_channels1, aggr='add')
#         # self.conv2 = GraphConv(hidden_channels1, hidden_channels2, aggr='add')
#         # # self.conv3 = GraphConv(hidden_channels2, hidden_channels3, aggr='add')
#         # # self.conv4 = GraphConv(hidden_channels3, hidden_channels4, aggr='add')
#         # self.lin1 = Linear(3, hidden_channels1)
#         # self.lin2 = Linear(hidden_channels2, 1)
#         # self.lin3 = Linear(hidden_channels6, 1)
#
#     def forward(self, x, edge_index, edge_index1):
#         x = self.edgeconv1(x, edge_index, edge_index1)
#         return x
#
#
# model = torch.load('GNN_try1.pkl')
# model.eval()
# if is_gpu:
#     model.cuda()
# else:
#     model.cpu()
#
# t3 = time.time()
# for data in data_list1:
#     x_cuda = data.x
#     edge_cuda = data.edge_index
#     edge_cuda1 = data.edge_index1
#     if is_gpu:
#         x_cuda = x_cuda.cuda()
#         edge_cuda = edge_cuda.cuda()
#         edge_cuda1 = edge_cuda1.cuda()
#         out = model(x_cuda, edge_cuda, edge_cuda1).cpu()
#     else:
#         out = model(x_cuda, edge_cuda, edge_cuda1)
#     # err = torch.linalg.norm(out - data.y) / torch.linalg.norm(data.y)
#     # print(out[0, :])
#     # print(data.y[0, :])
#     # print(f'error: {err:.6f}')
#
# t4 = time.time()
# time2 = (t4 - t3) / raw_data.shape[0]
# print(f'Calculation time: {time2:.6f}')
