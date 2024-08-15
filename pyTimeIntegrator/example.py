import pyTimeIntegrator as pyti
import numpy as np


def velocity_update(t, position):
    velocity = np.zeros(position.shape)
    for i in range(0, velocity.shape[0]):
        velocity[i, 0] = np.sin(t)
        velocity[i, 1] = np.sin(t)
        velocity[i, 2] = -np.sin(t)
        velocity[i, 3] = 1.0
    return velocity


Nc = 1000

# timeIntegrator = pyti.explicitRK4()
timeIntegrator = pyti.explicitEuler()
timeIntegrator.setTimeStep(0.1)
# timeIntegrator.setThreshold(1e-4)
timeIntegrator.setFinalTime(2000.0)
timeIntegrator.setOutputStep(1000)
timeIntegrator.setNumRigidBody(Nc)
timeIntegrator.setXLim([-0.5, 0.5])
timeIntegrator.setYLim([-0.5, 0.5])
timeIntegrator.setZLim([-0.5, 0.5])

position0 = np.zeros((Nc, 6))

timeIntegrator.initialize(position0)
timeIntegrator.setVelocityFunc(velocity_update)
timeIntegrator.run()
