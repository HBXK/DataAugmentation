#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:18:40 2024

@author: hk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation

# Constants
hbar = 1.0     # Reduced Planck constant (in arbitrary units)
m = 1.0        # Mass of the particle (in arbitrary units)
r = 1.0        # Radius in central potential
V0 = 1.0       # Step potential for theta > theta_0
theta_0 = np.pi / 4  # Boundary for step potential
N = 500        # Number of grid points in theta
theta_max = 2 * np.pi  # Max value of theta (mod 2pi)
theta_vals = np.linspace(0, theta_max, N)
dtheta = theta_vals[1] - theta_vals[0]  # Grid spacing in theta
dt = 0.001     # Time step size
t_max = 2.0    # Maximum time for simulation
n_steps = int(t_max / dt)

# Define the potential V(theta)
def potential(theta, V0, theta_0):
    V = np.zeros_like(theta)
    V[theta > theta_0] = V0
    return V

V = potential(theta_vals, V0, theta_0)

# Construct the finite difference matrix for d^2/dtheta^2
coeff = hbar**2 / (2 * m * r**2)
diagonals = [-2 * np.ones(N), np.ones(N-1), np.ones(N-1)]
D2 = diags(diagonals, [0, -1, 1], shape=(N, N)).toarray() / dtheta**2
D2[0, -1] = D2[-1, 0] = 1 / dtheta**2  # Periodic boundary conditions

# Initial wave packet (Gaussian)
def gaussian_wave_packet(theta, theta0, sigma):
    norm_factor = 1 / (sigma * np.sqrt(2 * np.pi))
    return norm_factor * np.exp(-(theta - theta0)**2 / (2 * sigma**2))

psi0 = gaussian_wave_packet(theta_vals, theta0=np.pi/2, sigma=0.1)
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dtheta)  # Normalize

# Crank-Nicolson matrices (A and B)
A = np.eye(N) + 1j * dt * (coeff * D2 + np.diag(V)) / (2 * hbar)
B = np.eye(N) - 1j * dt * (coeff * D2 + np.diag(V)) / (2 * hbar)

# Time evolution using Crank-Nicolson method
def crank_nicolson(psi0, A, B, n_steps):
    psi = psi0.copy()
    psi_t = np.zeros((n_steps, N), dtype=complex)
    psi_t[0, :] = psi0
    for n in range(1, n_steps):
        psi = spsolve(A, B @ psi)
        psi_t[n, :] = psi
    return psi_t

# Time evolution of the wavefunction
psi_t = crank_nicolson(psi0, A, B, n_steps)

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, theta_max)
ax.set_ylim(0, np.max(np.abs(psi_t)**2) * 1.1)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$|\psi(\theta, t)|^2$')
ax.axvline(theta_0, color='r', linestyle='--', label=r'$\theta_0$')
line, = ax.plot([], [], lw=2)

# Initialization function for the animation
def init():
    line.set_data([], [])
    return line,

# Update function for each frame
def update(frame):
    y_data = np.abs(psi_t[frame, :])**2
    line.set_data(theta_vals, y_data)
    ax.set_title(f'Time = {frame * dt:.3f}')
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True, interval=50)

# Display the animation
plt.legend()
plt.show()
