import numpy as np

from scipy import spatial


def Morse(r):
    r_norm = np.linalg.norm(r, axis=1)
    NN = r.shape[0]
    De = 1
    a = 1
    re = 2.5
    F_mag = -2 * a * De * (np.exp(-a * (r_norm - re)) -
                           np.exp(-2 * a * (r_norm - re)))
    F_mag = F_mag.reshape((NN, 1))
    r_norm = r_norm.reshape((NN, 1))
    F = r / r_norm * F_mag

    return F


def Add_ghost(x, xs, Lx, dx):
    x_ghost = np.copy(x)
    idx = np.where(x[:, 0] < xs + dx)
    temp = x[idx, :]
    temp = temp[0]
    temp[:, 0] = temp[:, 0] + Lx
    x_ghost = np.concatenate((x_ghost, temp), axis=0)
    idx = np.where(x[:, 0] > xs + Lx - dx)
    temp = x[idx, :]
    temp = temp[0]
    temp[:, 0] = temp[:, 0] - Lx
    x_ghost = np.concatenate((x_ghost, temp), axis=0)
    x = np.copy(x_ghost)
    idx = np.where(x[:, 1] < xs + dx)
    temp = x[idx, :]
    temp = temp[0]
    temp[:, 1] = temp[:, 1] + Lx
    x_ghost = np.concatenate((x_ghost, temp), axis=0)
    idx = np.where(x[:, 1] > xs + Lx - dx)
    temp = x[idx, :]
    temp = temp[0]
    temp[:, 1] = temp[:, 1] - Lx
    x_ghost = np.concatenate((x_ghost, temp), axis=0)
    x = np.copy(x_ghost)
    idx = np.where(x[:, 2] < xs + dx)
    temp = x[idx, :]
    temp = temp[0]
    temp[:, 2] = temp[:, 2] + Lx
    x_ghost = np.concatenate((x_ghost, temp), axis=0)
    idx = np.where(x[:, 2] > xs + Lx - dx)
    temp = x[idx, :]
    temp = temp[0]
    temp[:, 2] = temp[:, 2] - Lx
    x_ghost = np.concatenate((x_ghost, temp), axis=0)

    return x_ghost


def cal_Morse_force(x, Nc, R_cut, xs, Lx, dx):
    x = x.reshape(Nc, 3)
    x_ghost = Add_ghost(x, xs, Lx, dx)
    print(x_ghost.shape)
    F = np.zeros(x.shape)
    tree = spatial.cKDTree(x_ghost)
    edge = tree.query_ball_point(x, R_cut)
    num_edge = 0
    for i in range(Nc):
        idx1 = list(edge[i])
        num_edge = num_edge + len(edge[i])
        idx1.remove(i)
        F1 = Morse(x[i, :] - x_ghost[idx1, :])
        F[i, :] = np.sum(F1, axis=0)

    print(num_edge)

    return F
