#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 02:54:11 2025

@author: hk
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


k = 1.0          # Gravitational constant-like parameter
r0 = 1.0         # Initial radius
v0 = 0.8         # Initial tangential velocity
dt = 0.05        # Time step for the fixed-step methods (Euler, Verlet)
n_steps = 50000  # Number of steps for the fixed-step methods
m = 1.0          # mass = 1 for simplicity
# We will integrate from t=0 to t_max using all methods
t_max = n_steps * dt


r_init = np.array([r0, 0.0])
v_init = np.array([0.0, v0])

def potential(r):
    mag = np.linalg.norm(r)
    return -k / mag + 0.001 * np.cos(np.arctan(r[1]/r[0]))

def acceleration(pos):
    # a = -mdU
    r_vec = pos
    r_mag = np.linalg.norm(r_vec)
    return - k * r_vec / (r_mag**3) + 0.001*np.sin(np.arctan(pos[1]/pos[0]))

def total_energy(pos, vel):
    """
    E = T + V = 0.5*m*v^2 + V(r), with m=1 here.
    """
    r_mag = np.linalg.norm(pos)
    ke = 0.5 * np.dot(vel, vel)
    pe = potential(pos)
    return ke + pe


def euler_step(pos, vel, dt):

    a_n = acceleration(pos)
    pos_new = pos + dt * vel
    vel_new = vel + dt * a_n
    return pos_new, vel_new

def run_euler(pos_init, vel_init, dt, n_steps):
    pos = np.copy(pos_init)
    vel = np.copy(vel_init)
    times = []
    energies = []
    positions = []
    
    for i in range(n_steps+1):
        t = i * dt
        times.append(t)
        positions.append(pos)
        energies.append(total_energy(pos, vel))
        
        
        if i < n_steps:
            pos, vel = euler_step(pos, vel, dt)

    return np.array(times), np.array(positions), np.array(energies)


def velocity_verlet_step(pos, vel, dt):
    """
    Velocity Verlet:
      v_half = v_n + (dt/2)*a(pos_n)
      pos_{n+1} = pos_n + dt*v_half
      v_{n+1} = v_half + (dt/2)*a(pos_{n+1})
    """
    a_n = acceleration(pos)
    v_half = vel + 0.5*dt*a_n
    pos_new = pos + dt*v_half
    a_new = acceleration(pos_new)
    vel_new = v_half + 0.5*dt*a_new
    return pos_new, vel_new

def run_verlet(pos_init, vel_init, dt, n_steps):
    pos = np.copy(pos_init)
    vel = np.copy(vel_init)
    times = []
    energies = []
    positions = []
    J = []
    for i in range(n_steps+1):
        t = i * dt
        times.append(t)
        positions.append(pos)
        energies.append(total_energy(pos, vel))
        
        # Step
        if i < n_steps:
            pos, vel = velocity_verlet_step(pos, vel, dt)
            np.append(pos, 0)
            np.append(vel, 0)
            J.append(np.linalg.norm(m*np.cross(pos, vel)))
    return np.array(times), np.array(positions), np.array(energies), J


def kepler_deriv(t, y):

    x, y_, vx, vy = y
    pos = np.array([x, y_])
    a = acceleration(pos)
    return [vx, vy, a[0], a[1]]

def run_scipy_rk(pos_init, vel_init, t_max, n_points=50000):

    y0 = [pos_init[0], pos_init[1], vel_init[0], vel_init[1]]
    t_eval = np.linspace(0, t_max, n_points)
    
    sol = solve_ivp(kepler_deriv, [0, t_max], y0, t_eval=t_eval, method='RK45', max_step = 0.05,
                    rtol=1, atol=1)
    
    # Extract solution
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    vx_sol = sol.y[2]
    vy_sol = sol.y[3]
    times = sol.t
    J=[]
    # Compute energy at each output
    energies = []
    positions = []
    for i in range(len(times)):
        pos = np.array([x_sol[i], y_sol[i]])
        vel = np.array([vx_sol[i], vy_sol[i]])
        np.append(pos, 0)
        np.append(vel, 0)
        J.append(np.linalg.norm(m*np.cross(pos, vel)))
        energies.append(total_energy(pos, vel))
        positions.append(pos)
    
    return times, np.array(positions), np.array(energies), J


# Euler
t_euler, pos_euler, E_euler = run_euler(r_init, v_init, dt, n_steps)

# Velocity Verlet
t_verlet, pos_verlet, E_verlet, J_verlet = run_verlet(r_init, v_init, dt, n_steps)

# RK45 from scipy
t_rk, pos_rk, E_rk, J_rk = run_scipy_rk(r_init, v_init, t_max, n_points=n_steps+1)


# (A) ENERGY vs. TIME
plt.figure()
#plt.plot(t_euler, E_euler, label="Euler (non-symplectic)")
plt.plot(t_verlet, E_verlet, label="Velocity Verlet (symplectic)")
plt.plot(t_rk, E_rk, label="RK4")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.title("Energy vs. Time for 2D Kepler Problem (Perturbed)")
plt.legend()

# (B) ORBITAL TRAJECTORIES
plt.figure()
#plt.plot(pos_euler[:,0], pos_euler[:,1], label="Euler")
plt.plot(pos_verlet[:,0], pos_verlet[:,1], label="Velocity Verlet")
plt.plot(pos_rk[20000:28240,0], pos_rk[20000:28240,1], label="RK4")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbits in the Plane (-Euler) (Perturbed)")
plt.legend()


plt.figure()
#plt.plot(t_euler, E_euler, label="Euler (non-symplectic)")
plt.plot(t_verlet[1:], J_verlet, label="Velocity Verlet (symplectic)")
plt.plot(t_rk, J_rk, label="RK4")
plt.xlabel("Time")
plt.ylabel("Total Angular Momentum")
plt.title("Angular Momentum vs. Time for 2D Kepler Problem (Perturbed)")
plt.legend()


plt.show()
