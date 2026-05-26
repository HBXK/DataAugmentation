#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 23:06:46 2025

@author: hk
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y, eval_jacobi
import scipy.spatial.transform as sst
from sympy.physics.wigner import clebsch_gordan
from sympy import S
import sphericart as sc
from tqdm import tqdm
from math import factorial, sin, cos, sqrt
import sys

def sample_spheres_uniform(N, d):
    phi = 2*np.pi*np.random.uniform(0, 1, size=(N, d))
    
    cos_theta = np.random.uniform(0, 1, size=(N, d))
    theta = np.arccos(2*cos_theta-1)  

    points = np.stack((phi, theta), axis=-1)
    return points


def sph_to_cart(sph_angles):
 
    theta = sph_angles[..., 1]              # colatitude
    phi   = sph_angles[..., 0]              # azimuth
    sin_th = np.sin(theta)
    x = sin_th * np.cos(phi)
    y = sin_th * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)     # (..., 3, 3)


def cart_to_sph(cart):

    

    x, y, z = np.moveaxis(cart, -1, 0)      # unpack without extra copies

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)  
    phi = np.mod(phi, 2*np.pi)
    return np.stack((phi, theta), axis=-1)

def sample_SO3_via_QR(N):
    out = np.zeros((N,3,3))
    out[0, :] = np.eye(3)
    if N==1:
        return out
    for i in range(1,N):
        
        A = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(A)               
        if np.linalg.det(Q) < 0:             
            Q[:, 0] *= -1
        out[i, :] = Q
    return out




def gen_invar_basis_DM(deg, points):
    coeffs = []
    for l1 in range(deg+1):
        for l2 in range(deg+1-l1):
            for l3 in range(abs(l1-l2),l1+l2+1):
                if l1+l2+l3 <= deg:
                    coeffs.append([l1,l2,l3])
    Basis = np.zeros((len(coeffs),len(points)), dtype='complex128')            
    for i,L in enumerate(coeffs):
        
        for m1 in range(-l1,l1+1):
            for m2 in range(-l2,l2+1):
                m3 = -m1-m2
                if -L[2]<= m3 <= L[2]:
                    C00 = float(clebsch_gordan(S(L[2]), S(L[2]), 0, S(m3), S(-m3), 0))
                    Cmm = float(clebsch_gordan(S(L[0]), S(L[1]), S(L[2]), S(m1), S(m2), S(-m3)))
                    Basis[i] += C00*Cmm * sph_harm_y(L[0],m1,points[::, 0, 1], points[::, 0, 0]) * sph_harm_y(L[1],m2,points[::, 1, 1],points[::, 1, 0]) * sph_harm_y(L[0],m3,points[::, 2, 1],points[::, 2, 0])
    return Basis


point = sample_spheres_uniform(10, 3)

R = sample_SO3_via_QR(2)[1]

test = gen_invar_basis_DM(2,point)
sym_test = gen_invar_basis_DM(2, cart_to_sph(sph_to_cart(point)@(R.T)))

print(test-sym_test)