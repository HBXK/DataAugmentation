#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 03:35:53 2025

@author: hk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------
# 1) PROBLEM SETUP
# --------------------------------------------------------------------

G = 1.0  # Gravitational constant (choose units so that G=1)
m = np.array([10000,20,2])  # masses of the three bodies

# You can pick your own initial conditions. Here's a simple setup:
# Let's put them in roughly triangular positions with some velocities.
# Positions (x,y) and velocities (vx, vy) for each body i:
r_init = np.array([
    [ 0.0,  0.0],   # Body 1
    [0,  500],  # Body 2
    [0, -1000]   # Body 3
])
v_init = np.array([
    [ 0.1,  0],   # Body 1 velocity
    [ 3, 3],  # Body 2 velocity
    [-4, 1]   # Body 3 velocity
])

# Time parameters
dt = 0.005   # Time step for Velocity Verlet
n_steps = 300000  # Number of steps
t_max = n_steps * dt  # Final time


# --------------------------------------------------------------------
# 2) HELPER FUNCTIONS
# --------------------------------------------------------------------
def A(r):
    A = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i==j:
                continue
            else:
                A[i,j]=G*m[j]/np.linalg.norm(r[i]-r[j],2)
    diag_mask = np.eye(A.shape[0], dtype=bool)

    # Invert the mask (to select off-diagonal only) and reshape to 3×2
    off_diag = A[~diag_mask].reshape(3, 2)
    return off_diag

def accelerations(positions):
    """
    Compute the 2D acceleration on each body due to the other bodies.
    positions: array of shape (3, 2) for 3 bodies in 2D.
    returns: array of shape (3, 2) of accelerations.
    """
    acc = np.zeros_like(positions)  # (3, 2)
    for i in range(3):
        # Sum over j != i
        for j in range(3):
            if j == i:
                continue
            diff = positions[i] - positions[j]
            r3 = np.linalg.norm(diff)**3
            acc[i] += -G * m[j] * diff / r3
    return acc

def total_energy(positions, velocities):
    """
    Total energy = Kinetic + Potential.
    Kinetic = 0.5 * sum_i(m_i * v_i^2).
    Potential = sum_{i<j} -G*m_i*m_j / |r_i - r_j|.
    positions, velocities: shape (3, 2).
    """
    # Kinetic
    KE = 0.0
    for i in range(3):
        KE += 0.5 * m[i] * np.dot(velocities[i], velocities[i])
    # Potential
    PE = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)
            PE += -G * m[i] * m[j] / dist
    return KE + PE

# --------------------------------------------------------------------
# 3) VELOCITY VERLET INTEGRATOR
# --------------------------------------------------------------------
def velocity_verlet(r0, v0, dt, n_steps):
    """
    r0, v0: initial (positions, velocities) arrays of shape (3,2)
    dt: time step
    n_steps: number of steps
    returns times, positions, energies
      where positions is shape (n_steps+1, 3, 2)
    """
    r = np.copy(r0)
    v = np.copy(v0)

    # Arrays to store data
    times = np.zeros(n_steps+1)
    positions = np.zeros((n_steps+1, 3, 2))
    energies = np.zeros(n_steps+1)

    # Initial data
    positions[0] = r
    energies[0] = total_energy(r, v)

    a = accelerations(r)  # initial acceleration
    for i in range(n_steps):
        # Half-step velocity
        v_half = v + 0.5*dt*a
        
        # Full-step position
        r_new = r + dt*v_half
        
        # Acceleration at new position
        a_new = accelerations(r_new)
        
        # Complete velocity update
        v_new = v_half + 0.5*dt*a_new
        
        # Save
        r = r_new
        v = v_new
        a = a_new
        
        positions[i+1] = r
        energies[i+1] = total_energy(r, v)
        times[i+1] = (i+1)*dt

    return times, positions, energies

def stormer_verlet(r0, v0, dt, n_steps):
    times = np.zeros(n_steps+1)
    positions = np.zeros((n_steps+1, 3, 2))
    energies = np.zeros(n_steps+1)
    positions[0] = r0
    energies[0] = total_energy(r0, v0)
    positions[1]=r0+v0*dt+0.5*A(r0)*dt**2
    times[1]=dt
    for t in range(2,n_steps+1):
        positions[t]=2*positions[t-1]-positions[t-2]+A(positions[t-1])*dt**2
        v = (positions[t]-positions[t-1])/dt
        energies[t] = total_energy(positions[t],v)
        times[t]=(t)*dt
    return times, positions, energies
# --------------------------------------------------------------------
# 4) RUNGE-KUTTA (solve_ivp)
# --------------------------------------------------------------------
def three_body_ode(t, y):
    """
    ODE function for 3 bodies in 2D.
    y has length 12: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
    Returns dy/dt in same 12D shape.
    """
    # Unpack
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y
    
    pos = np.array([[x1, y1],
                    [x2, y2],
                    [x3, y3]])
    vel = np.array([[vx1, vy1],
                    [vx2, vy2],
                    [vx3, vy3]])
    # Acceleration
    acc = accelerations(pos)  # shape (3, 2)

    # Pack derivatives
    dydt = [
        vx1, vy1,  # dx1/dt, dy1/dt
        vx2, vy2,
        vx3, vy3,
        acc[0,0], acc[0,1],  # dvx1/dt, dvy1/dt
        acc[1,0], acc[1,1],
        acc[2,0], acc[2,1]
    ]
    return dydt

def run_scipy_rk(r0, v0, t_max, n_save=2001):
    """
    Integrate the 3-body problem with RK45 from scipy.
    We'll sample the solution at n_save points between 0 and t_max.
    """
    # Pack initial state
    y0 = [
        r0[0,0], r0[0,1],
        r0[1,0], r0[1,1],
        r0[2,0], r0[2,1],
        v0[0,0], v0[0,1],
        v0[1,0], v0[1,1],
        v0[2,0], v0[2,1]
    ]
    t_eval = np.linspace(0, t_max, n_save)
    
    sol = solve_ivp(three_body_ode, [0, t_max], y0, t_eval=t_eval, method='RK45',
                    rtol=1e-9, atol=1e-12)
    
    # Extract
    times = sol.t
    y = sol.y.T  # shape (n_save, 12)
    
    # Reformat
    positions = []
    energies = []
    for row in y:
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = row
        pos = np.array([[x1, y1],
                        [x2, y2],
                        [x3, y3]])
        vel = np.array([[vx1, vy1],
                        [vx2, vy2],
                        [vx3, vy3]])
        positions.append(pos)
        energies.append(total_energy(pos, vel))
    return times, np.array(positions), np.array(energies)

# --------------------------------------------------------------------
# 5) RUN BOTH METHODS
# --------------------------------------------------------------------
# Velocity Verlet
t_verlet, pos_verlet, E_verlet = velocity_verlet(r_init, v_init, dt, n_steps)

# RK45 (we'll sample about as many points to compare)
t_rk, pos_rk, E_rk = run_scipy_rk(r_init, v_init, t_max, n_save=3001)

# --------------------------------------------------------------------
# 6) PLOTS
# --------------------------------------------------------------------
plt.figure()
plt.plot(t_verlet, E_verlet, label="Velocity Verlet (Symplectic)")
plt.plot(t_rk, E_rk, label="RK45 (solve_ivp)")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.title("3-Body Problem: Energy vs. Time")
plt.legend()

# Let's plot the orbits in the XY-plane. We'll show just the final positions
# or selected trajectories for each method.
plt.figure()
# Verlet orbits
plt.plot(pos_verlet[:,0,0], pos_verlet[:,0,1], label="Body1-Verlet")
plt.plot(pos_verlet[:,1,0], pos_verlet[:,1,1], label="Body2-Verlet")
plt.plot(pos_verlet[:,2,0], pos_verlet[:,2,1], label="Body3-Verlet")

# RK orbits
plt.plot(pos_rk[:,0,0], pos_rk[:,0,1], '--', label="Body1-RK")
plt.plot(pos_rk[:,1,0], pos_rk[:,1,1], '--', label="Body2-RK")
plt.plot(pos_rk[:,2,0], pos_rk[:,2,1], '--', label="Body3-RK")

plt.xlabel("x")
plt.ylabel("y")
plt.title("3-Body Orbits in 2D")
plt.legend()
plt.show()
