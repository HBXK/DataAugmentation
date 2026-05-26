#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:29:53 2024

@author: hk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


# Constants
hbar = 1.0     # Reduced Planck constant (in arbitrary units)
m = 1.0        # Mass of the particle (in arbitrary units)
V0 = 1.0       # Step potential for theta > theta_0
theta_0 = np.pi / 4  # Boundary for step potential
omega = 1.0    # Frequency for harmonic potential
r_min, r_max = 0.5, 5.0  # Min and max radius
N_r = 200      # Number of grid points in r
N_theta = 500  # Number of grid points in theta
r_vals = np.linspace(r_min, r_max, N_r)
theta_vals = np.linspace(0, 2 * np.pi, N_theta)
dr = r_vals[1] - r_vals[0]   # Grid spacing in r
dtheta = theta_vals[1] - theta_vals[0]  # Grid spacing in theta
dt = 0.001     # Time step size
t_max = 2.0    # Maximum time for simulation
n_steps = int(t_max / dt)

# Define the step potential V(theta)
def step_potential(theta, V0, theta_0):
    V = np.zeros_like(theta)
    V[theta > theta_0] = V0
    return V

V_theta = step_potential(theta_vals, V0, theta_0)

# Define the central harmonic potential V(r) = 1/2 * m * omega^2 * r^2
def harmonic_potential(r, omega, m):
    return 0.5 * m * omega**2 * r**2

V_r = harmonic_potential(r_vals, omega, m)

# Create a 2D potential grid V(r, theta)
V = np.zeros((N_r, N_theta))
for i in range(N_r):
    V[i, :] = V_r[i] + V_theta

# Construct finite difference matrix for r
coeff_r = hbar**2 / (2 * m)
diagonals_r = [-2 * np.ones(N_r), np.ones(N_r-1), np.ones(N_r-1)]
D2_r = diags(diagonals_r, [0, -1, 1], shape=(N_r, N_r)).toarray() / dr**2
D2_r[0, 1] = D2_r[1, 0]  # Apply boundary conditions for r

# Construct finite difference matrix for theta
coeff_theta = hbar**2 / (2 * m)
D2_theta = np.zeros((N_theta, N_theta))
for j in range(N_theta):
    D2_theta[j, j] = -2.0
    D2_theta[j, (j + 1) % N_theta] = 1.0  # Right neighbor
    D2_theta[j, (j - 1) % N_theta] = 1.0  # Left neighbor
D2_theta /= dtheta**2  # Normalize for spacing

# Initial Gaussian wave packet in both r and theta
def gaussian_wave_packet(r, theta, r0, theta0, sigma_r, sigma_theta):
    norm_factor_r = 1 / (sigma_r * np.sqrt(2 * np.pi))
    norm_factor_theta = 1 / (sigma_theta * np.sqrt(2 * np.pi))
    return norm_factor_r * np.exp(-(r - r0)**2 / (2 * sigma_r**2)) * \
           norm_factor_theta * np.exp(-(theta - theta0)**2 / (2 * sigma_theta**2))

psi0 = np.zeros((N_r, N_theta), dtype=complex)
for i in range(N_r):
    psi0[i, :] = gaussian_wave_packet(r_vals[i], theta_vals, r0=3.0, theta0=np.pi/2, sigma_r=0.2, sigma_theta=0.1)
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dr * dtheta)  # Normalize

# Crank-Nicolson matrices (A and B for both r and theta)
A_r = np.eye(N_r) + 1j * dt * coeff_r * D2_r / (2 * hbar)
B_r = np.eye(N_r) - 1j * dt * coeff_r * D2_r / (2 * hbar)
A_theta = np.eye(N_theta) + 1j * dt * coeff_theta * D2_theta / (2 * hbar)
B_theta = np.eye(N_theta) - 1j * dt * coeff_theta * D2_theta / (2 * hbar)

# Time evolution using Crank-Nicolson method
def crank_nicolson_2d(psi0, A_r, B_r, A_theta, B_theta, V, n_steps):
    psi = psi0.copy()
    psi_t = np.zeros((n_steps, N_r, N_theta), dtype=complex)
    psi_t[0, :, :] = psi0
    for n in tqdm(range(1, n_steps)):
        for i in range(N_r):
            psi[i, :] = spsolve(A_theta, B_theta @ psi[i, :])  # Theta evolution
        for j in range(N_theta):
            psi[:, j] = spsolve(A_r, B_r @ psi[:, j])  # Radial evolution
        psi *= np.exp(-1j * V * dt / hbar)  # Apply potential term
        psi_t[n, :, :] = psi
    return psi_t

# Time evolution of the wavefunction
psi_t = crank_nicolson_2d(psi0, A_r, B_r, A_theta, B_theta, V, n_steps)

# Plot setup for animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, np.max(np.abs(psi_t)**2) * 1.1)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$|\psi(r, \theta, t)|^2$')
line, = ax.plot([], [], lw=2)

# Initialization function for the animation
def init():
    line.set_data([], [])
    return line,

# Update function for each frame
def update(frame):
    y_data = (np.abs(psi_t[frame, :, :])**2).sum(axis=0)  # Sum over r to get theta-projection
    line.set_data(theta_vals, y_data)
    ax.set_title(f'Time = {frame * dt:.3f}')
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True, interval=50)

# Display the animation
plt.show()

