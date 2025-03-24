import numpy as np
import cupy as cp  # GPU acceleration
from numpy import cos, sin
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Constants
G = 9.8
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0
t_stop = 10.0  # Extended time
dt = 0.001  # Higher resolution
t = np.arange(0, t_stop, dt)

# Initial Conditions
th1, w1, th2, w2 = 120.0, 0.0, -10.0, 0.0
state = np.radians([th1, w1, th2, w2])

# Function to Compute Derivatives
def derivs(t, state):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    
    delta = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta) ** 2
    dydx[1] = ((M2 * L1 * state[1] ** 2 * sin(delta) * cos(delta) +
                 M2 * G * sin(state[2]) * cos(delta) +
                 M2 * L2 * state[3] ** 2 * sin(delta) -
                 (M1 + M2) * G * sin(state[0])) / den1)
    
    dydx[2] = state[3]
    
    den2 = (L2 / L1) * den1
    dydx[3] = ((-M2 * L2 * state[3] ** 2 * sin(delta) * cos(delta) +
                 (M1 + M2) * G * sin(state[0]) * cos(delta) -
                 (M1 + M2) * L1 * state[1] ** 2 * sin(delta) -
                 (M1 + M2) * G * sin(state[2])) / den2)
    
    return dydx

# Integrate Using Euler's Method
y = np.empty((len(t), 4))
y[0] = state
for i in range(1, len(t)):
    y[i] = y[i - 1] + derivs(t[i - 1], y[i - 1]) * dt

# Store as GPU Array
y_gpu = cp.asarray(y)

# Extract Features for Clustering
features = np.column_stack([y[:, 0], y[:, 1], y[:, 2], y[:, 3]])

# Perform DBSCAN Clustering
clustering = DBSCAN(eps=0.1, min_samples=10).fit(features)
labels = clustering.labels_

# Plot Clusters
plt.scatter(y[:, 0], y[:, 2], c=labels, cmap='viridis', s=1)
plt.xlabel("Theta 1 (rad)")
plt.ylabel("Theta 2 (rad)")
plt.title("Clustering of Double Pendulum Dynamics")
plt.show()
