import sys
import numpy as np
from scipy.integrate import odeint
import os
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
from multiprocessing import Pool

# Pendulum parameters
L1, L2 = 1, 1  # Rod lengths (m)
m1, m2 = 1, 1  # Bob masses (kg)
g = 9.81  # Gravitational acceleration (m.s^-2)

def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def calc_E(y):
    """Return the total energy of the system."""
    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

# Ask user for training time
time_input = input("Enter training time (e.g., 3600s or 2h): ")
if time_input.endswith('h'):
    tmax = int(time_input[:-1]) * 3600  # Convert hours to seconds
elif time_input.endswith('s'):
    tmax = int(time_input[:-1])
else:
    print("Invalid input format. Use 'xs' for seconds or 'xh' for hours.")
    sys.exit()
file_path = f"pendulum_data{tmax}.npz"

# Check if file already exists
if os.path.exists(file_path):
    print(f"File {file_path} already exists. Choose a different training time.")
    sys.exit()

# Time settings
dt = 0.001
t = np.arange(0, tmax+dt, dt)

# Initial conditions
y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])

# Data storage
chunk_size = 1000
chunks = [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]

def process_chunk(chunk_y0_chunk):
    chunk, y0 = chunk_y0_chunk
    y_chunk = odeint(deriv, y0, chunk, args=(L1, L2, m1, m2))
    theta1, theta2 = y_chunk[:, 0], y_chunk[:, 2]
    x1, y1 = L1 * np.sin(theta1), -L1 * np.cos(theta1)
    x2, y2 = x1 + L2 * np.sin(theta2), y1 - L2 * np.cos(theta2)
    E = calc_E(y_chunk)
    return chunk, theta1, theta2, x1, y1, x2, y2, E, y_chunk[-1]

# Use multiprocessing for parallel computation
num_cores_to_use = max(1, os.cpu_count() - 10)  # Adjust to save 2 cores
with Pool(num_cores_to_use) as pool:
    results = list(tqdm(pool.imap(process_chunk, [(chunk, y0) for chunk in chunks]), total=len(chunks), desc="Integrating", unit="chunk"))

time_data, theta1_data, theta2_data = [], [], []
x1_data, y1_data, x2_data, y2_data, energy_data = [], [], [], [], []

for chunk, theta1, theta2, x1, y1, x2, y2, E, y0 in results:
    time_data.append(chunk)
    theta1_data.append(theta1)
    theta2_data.append(theta2)
    x1_data.append(x1)
    y1_data.append(y1)
    x2_data.append(x2)
    y2_data.append(y2)
    energy_data.append(E)
print("saving...")
# Save data
np.savez(file_path, Time=np.concatenate(time_data), Theta1=np.concatenate(theta1_data),
         Theta2=np.concatenate(theta2_data), X1=np.concatenate(x1_data), Y1=np.concatenate(y1_data),
         X2=np.concatenate(x2_data), Y2=np.concatenate(y2_data), Energy=np.concatenate(energy_data))

print(f"Data saved to {file_path}")

# Load and print the saved data
print("loading data...")
loaded_data = np.load(file_path)
print("reading data...")
df = pd.DataFrame({key: loaded_data[key] for key in loaded_data.keys()})
print(tabulate(df.head(), headers='keys', tablefmt='psql'))
