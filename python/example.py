import os
import hignn
import numpy as np
import sys
from mpi4py import MPI

def velocity_update(t, position):
    if rank == 0:
        print("\n t = ", t)
    hignn_model.UpdateCoord(position)
    hignn_model.Update()
    velocity = np.zeros(position.shape)
    force = np.ones(position.shape)
    
    hignn_model.Dot(velocity, force)
    
    return velocity

if __name__ == '__main__':
    os.system('clear')
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()

    N = 15
    nx = N
    ny = N
    nz = N
    dx = 3
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dx, dx)
    z = np.arange(0, nz * dx, dx)
    xx, yy, zz = np.meshgrid(x, y, z)
    X = np.concatenate(
        (xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    X = X.astype(np.float32)
    
    NN = nx * ny * nz

    np.random.seed(0)

    X += 0.2 * np.random.rand(nx * ny * nz, dx)

    hignn_model = hignn.HignnModel(X, 100)
    
    hignn_model.LoadTwoBodyModel('nn/3D_force_UB_max600_try2')
    
    # set parameters for far dot, the following parameters are default values
    hignn_model.SetEpsilon(0.05)
    hignn_model.SetMaxIter(50)
    hignn_model.SetMatPoolSizeFactor(40)
    hignn_model.SetPostCheckFlag(True)
    
    time_integrator = hignn.ExplicitEuler()
    
    time_integrator.setTimeStep(0.001)
    time_integrator.setFinalTime(0.001)
    time_integrator.setNumRigidBody(NN)
    time_integrator.setOutputStep(1)
    time_integrator.initialize(X)
    time_integrator.setVelocityFunc(velocity_update)
    time_integrator.run()
    
    edgeInfo = hignn.BodyEdgeInfo()
    edgeInfo.setThreeBodyEpsilon(5.0)

    edgeInfo.setTargetSites(X)

    edgeInfo.buildThreeBodyEdgeInfo()
    
    del hignn_model
    del edgeInfo
    
    hignn.Finalize()