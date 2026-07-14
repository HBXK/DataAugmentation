#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 13:10:53 2025

@author: hk
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y, eval_jacobi
import scipy.spatial.transform as sst
from sympy.physics.wigner import clebsch_gordan
from sympy import S
from tqdm import tqdm
from math import factorial, sin, cos, sqrt
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split

from scipy.stats import special_ortho_group



def sample_spheres_uniform(N, d):
    phi = 2*np.pi*np.random.uniform(0, 1, size=(N, d))
    
    cos_theta = np.random.uniform(0, 1, size=(N, d))
    theta = np.arccos(2*cos_theta-1)  

    points = np.stack((phi, theta), axis=-1)
    return points


def sphere_1_constraint(N,d):
    points = sample_spheres_uniform(N,d)
    points[:, 0, :] = 0
    return points

def sphere_2_constraint(N,d):
    points = sample_spheres_uniform(N,d)
    points[:, 0, :] = 0
    points[:, 1, 0] = np.pi/2
    return points

def sphere_1_const_pert(N,d, eps=1e-1):
    points = sphere_1_constraint(N,d)
    points[:,0,0] = np.mod(np.random.normal(0, eps, size=N), 2*np.pi)
    points[:,0,1] = np.abs(np.random.normal(0,eps, size=N))
    return points

def sphere_2_const_pert(N,d, eps=1e-1):
    points = sphere_2_constraint(N,d)
    points[:,0,0] = np.mod(np.random.normal(0, eps, size=N), 2*np.pi)
    points[:,0,1] = np.abs(np.random.normal(0,eps, size=N))
    points[:,1,0] += np.random.normal(0,eps, size=N)
    points[:,1,1] += np.random.normal(0, eps, size=N)
    points[:,1,1] = np.abs(points[:,1,1])
    return points





def eval_invar_sph_basis_3D(l1, l2, l3, m1, m2, m3, points):
    if m1 + m2 + m3 != 0:
        return np.zeros(points.shape[0], dtype=np.complex128)

    N = points.shape[0]
    result = np.zeros(N, dtype=np.complex128)

    phi1, theta1 = points[:, 0, 0], points[:, 0, 1]
    phi2, theta2 = points[:, 1, 0], points[:, 1, 1]
    phi3, theta3 = points[:, 2, 0], points[:, 2, 1]

    L_min = abs(l1 - l2)
    L_max = l1 + l2


    for mu1 in range(-l1, l1+1):
        for mu2 in range(-l2, l2+1):
            mu3 = -mu2 -mu1
            if -l3<= mu3<=l3 and L_min <= l3 <= L_max:
                L = l3
                
                # C_mm = float(clebsch_gordan(S(l1), S(m1), S(l2), S(m2), S(L), S(m1 + m2)))
                # C_mu = float(clebsch_gordan(S(l1), S(mu1), S(l2), S(mu2), S(L), S(mu1 + mu2)))
                    
                # C_mu3 = float(clebsch_gordan(S(L), S(mu1 + mu2), S(l3), S(mu3), S(0), S(0)))
                # C_mm3 = float(clebsch_gordan(S(L), S(m1 + m2), S(l3), S(m3), S(0), S(0)))
                
                C_mm = float(clebsch_gordan(S(l1), S(l2), S(L), S(m1), S(m2), S(m1 + m2)))
                C_mu = float(clebsch_gordan(S(l1), S(l2), S(L), S(mu1), S(mu2),  S(mu1 + mu2)))
            
                C_mu3 = float(clebsch_gordan(S(L), S(l3), S(0), S(mu1 + mu2), S(mu3), S(0)))
                C_mm3 = float(clebsch_gordan(S(L), S(l3), S(0),S(m1 + m2), S(m3), S(0)))
                
                   
                coeff = C_mm * C_mu * C_mm3 * C_mu3
                Y1 = sph_harm_y(l1, mu1, theta1, phi1)
                Y2 = sph_harm_y(l2, mu2, theta2, phi2)
                Y3 = sph_harm_y(l3, mu3, theta3, phi3)

                result += coeff * Y1 * Y2 * Y3
                


    return result


    


def evaluate_sum_of_invar_basis_3D(points, coeffs, degree):
    """
    Evaluate the function f(points) = sum c_{l1l2l3m1m2m3} * B_{l1l2l3}^{m1m2m3}(points)

    Parameters:
        points: ndarray of shape (N, 3, 2)
        coeffs: dict with keys (l1, l2, l3, m1, m2, m3) and real values
        degree: int, upper bound on l1 + l2 + l3

    Returns:
        values: ndarray of shape (N,) with complex values
    """
    N = points.shape[0]
    result = np.zeros(N, dtype=np.complex128)

    for (l1, l2, l3, m1, m2, m3), c_val in tqdm(coeffs.items()):
        if l1 + l2 + l3 > degree:
            continue
        if m1 + m2 + m3 != 0:
            continue
        basis_val = eval_invar_sph_basis_3D(l1, l2, l3, m1, m2, m3, points)
        result += c_val * basis_val

    return result



def generate_invar_basis_keys_3D(degree):
    """
    Generate constrained basis keys with m1 + m2 + m3 = 0 and l1+l2+l3 <= degree.
    
    Returns:
        keys: list of (l1, l2, l3, m1, m2, m3)
    """
    keys = []
    for l1 in range(degree + 1):
        for l2 in range(degree + 1 - l1):
            for l3 in range(degree + 1 - l1 - l2):
                if abs(l1-l2)<=l3<=l1+l2:

                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            m3 = -(m1 + m2)
                            if -l3 <= m3 <= l3:
                                keys.append((l1, l2, l3, m1, m2, m3))
    return keys

def eval_invar_design_matrix_3D(points, degree):
    """
    Construct the design matrix for the invariant spherical harmonic basis.
    
    Parameters:
        points: ndarray of shape (N, 3, 2)
        degree: int
    
    Returns:
        Φ: ndarray of shape (N, B)
        keys: list of (l1, l2, l3, m1, m2, m3)
    """
    keys = generate_invar_basis_keys_3D(degree)
    N = points.shape[0]
    B = len(keys)
    Φ = np.zeros((N, B), dtype=np.complex128)

    for j, (l1, l2, l3, m1, m2, m3) in enumerate(keys):
        Φ[:, j] = eval_invar_sph_basis_3D(l1, l2, l3, m1, m2, m3, points)

    return Φ, keys

def fit_invar_least_squares_3D(points, values, degree):

    
    Φ, keys = eval_invar_design_matrix_3D(points, degree)  

    coeff_vector, *_ = np.linalg.lstsq(Φ, values, rcond=None)  

    coeffs = {tuple(key): coeff_vector[i] for i, key in enumerate(keys)}
    return coeffs    



def evaluate_invar_least_squares_3D(points_new, coeffs, degree):

    Φ_new, keys =  eval_invar_design_matrix_3D(points_new, degree)

    coeff_vector = np.array([coeffs.get(tuple(k), 0.0) for k in keys], dtype=np.complex128)

    return Φ_new @ coeff_vector  




def generate_basis_keys_vectorized_3D(degree):
    """
    Generate (l1, l2, l3, m1, m2, m3) with l1 + l2 + l3 <= degree

    Returns:
        keys: ndarray of shape (B, 6)
    """
    l_triplets = []
    for l1 in range(degree + 1):
        for l2 in range(degree + 1 - l1):
            # for l3 in range(np.abs(l1-l2),l1+l2+1):
            #     if l1+l2+l3<=degree:
            #         l_triplets.append((l1, l2, l3))
            for l3 in range(degree+1-l1-l2):
                l_triplets.append((l1,l2,l3))
                            

    key_list = []
    for l1, l2, l3 in l_triplets:
        m1_vals = np.arange(-l1, l1 + 1)
        m2_vals = np.arange(-l2, l2 + 1)
        m3_vals = np.arange(-l3, l3 + 1)

        M1, M2, M3 = np.meshgrid(m1_vals, m2_vals, m3_vals, indexing='ij')
        M1 = M1.flatten()
        M2 = M2.flatten()
        M3 = M3.flatten()

        n = len(M1)

        keys = np.zeros((n, 6), dtype=int)
        keys[:, 0] = l1
        keys[:, 1] = l2
        keys[:, 2] = l3
        keys[:, 3] = M1
        keys[:, 4] = M2
        keys[:, 5] = M3

        key_list.append(keys)

    return np.vstack(key_list)

def gen_coeffs_vectorized_3D(degree):
    """
    Generate coefficients c_{l1 l2 l3 m1 m2 m3} 

    Returns:
        coeffs: dict {(l1, l2, l3, m1, m2, m3): complex}
    """
    keys = generate_basis_keys_vectorized_3D(degree)
    coeffs = {}

    for key in keys:
        l1, l2, l3, m1, m2, m3 = key
        decay = np.exp(-2*(l1 + l2 + l3))
        real = np.random.uniform(-1, 1)
        coeffs[tuple(key)] = (real) * decay

    return coeffs


def gen_invar_coeffs_vectorized_3D(degree):
    """
    Generate coefficients c_{l1 l2 l3 m1 m2 m3} 

    Returns:
        coeffs: dict {(l1, l2, l3, m1, m2, m3): complex}
    """
    keys = generate_invar_basis_keys_3D(degree)
    coeffs = {}

    for key in keys:
        l1, l2, l3, m1, m2, m3 = key
        decay = np.exp(-(l1 + l2 + l3))
        real = np.random.uniform(-1, 1)
        coeffs[tuple(key)] = real * decay

    return coeffs




def eval_design_matrix_vectorized_3D(points, degree):
    """
    Evaluate design matrix using constrained spherical harmonic triple basis.

    Parameters:
        points: ndarray (N, 3, 2)
        degree: int

    Returns:
        Φ: ndarray (N, B)
        keys: ndarray (B, 6)
    """
    N = points.shape[0]
    keys = generate_basis_keys_vectorized_3D(degree)

    # Precompute spherical harmonics for each point and each (l, m)
    Y = [{}, {}, {}]
    for i in range(3):
        phi = points[:, i, 0]
        theta = points[:, i, 1]
        for l in range(degree + 1):
            for m in range(-l, l + 1):
                Y[i][(l, m)] = sph_harm_y(l, m, theta, phi)

    # Evaluate all basis functions
    B = len(keys)
    Φ = np.empty((N, B), dtype=np.complex128)
    for j, (l1, l2, l3, m1, m2, m3) in enumerate(keys):
        Φ[:, j] = Y[0][(l1, m1)] * Y[1][(l2, m2)] * Y[2][(l3, m3)]

    return Φ, keys



def eval_proj_matrix(degree):
    keys = generate_invar_basis_keys_3D(degree)
    P = np.zeros((len(keys), 2*degree + 1))
    for j, (l1, l2, l3, m1, m2, m3) in enumerate(keys):
        L_min = abs(l1 - l2)
        L_max = l1 + l2
        for mu1 in range(-l1, l1+1):
            for mu2 in range(-l2, l2+1):
                mu3 = -mu2 -mu1
                if -l3<= mu3<=l3 and L_min <= l3 <= L_max:
                    L = l3
                    
 
                    C_mm = float(clebsch_gordan(S(l1), S(l2), S(L), S(m1), S(m2), S(m1 + m2)))
                    C_mu = float(clebsch_gordan(S(l1), S(l2), S(L), S(mu1), S(mu2),  S(mu1 + mu2)))
                        
                    C_mu3 = float(clebsch_gordan(S(L), S(l3), S(0), S(mu1 + mu2), S(mu3), S(0)))
                    C_mm3 = float(clebsch_gordan(S(L), S(l3), S(0),S(m1 + m2), S(m3), S(0)))
                    
                       
                    P[j, degree + mu1 + mu2] += C_mm * C_mu * C_mm3 * C_mu3
    Q,R = np.linalg.qr(P)
    return P # Q@Q.T



def fit_least_squares_3D(points, values, degree, tol):

    
    Φ, keys = eval_design_matrix_vectorized_3D(points, degree)  

    coeff_vector, *_ = np.linalg.lstsq(Φ, values, rcond=tol)  

    coeffs = {tuple(key): coeff_vector[i] for i, key in enumerate(keys)}
    return coeffs    


def evaluate_full_least_squares_3D(points_new, coeffs, degree):

    Φ_new, keys = eval_design_matrix_vectorized_3D(points_new, degree)

    coeff_vector = np.array([coeffs.get(tuple(k), 0.0) for k in keys], dtype=np.complex128)

    return Φ_new @ coeff_vector  

def read_so3_quadrature_file(file_path):

    matrices = []
    weights = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines[2:]:
            values = list(map(float, line.split()))
            
            matrix = np.array(values[:9]).reshape((3, 3))
            matrices.append(matrix)
            
            weight = values[9]
            weights.append(weight)
    
    return matrices, weights

def sph_to_cart(sph_angles):
    """
    Convert an (..., 3, 2) block of (theta, phi) angles to (..., 3, 3) Cartesian coords.

    Parameters
    ----------
    sph_angles : ndarray
        Angles in radians, last dimension is (theta, phi).
    r : float, optional
        Sphere radius, default 1.0.

    Returns
    -------
    cart : ndarray
        Cartesian coordinates with shape sph_angles.shape[:-1] + (3,)
    """
    theta = sph_angles[..., 1]              # colatitude
    phi   = sph_angles[..., 0]              # azimuth
    sin_th = np.sin(theta)
    x = sin_th * np.cos(phi)
    y = sin_th * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)     # (..., 3, 3)


def cart_to_sph(cart):
    """
    Convert an (..., 3, 3) block of (x, y, z) to (..., 3, 2) spherical angles (theta, phi).

    Parameters
    ----------
    cart : ndarray
        Cartesian coordinates, last dimension is (x, y, z).

    Returns
    -------
    sph_angles : ndarray
        Same leading shape as `cart`, last dimension (theta, phi) in radians.
        θ ∈ [0, π], φ ∈ (–π, π].
    """

    x, y, z = np.moveaxis(cart, -1, 0)      # unpack without extra copies

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)  
    phi = np.mod(phi, 2*np.pi)
    return np.stack((phi, theta), axis=-1)  # (..., 3, 2)
    
    
def quad_augmented_design_matrix(points, degree, rotation_matrices, w):

    augmented_matrices = []
    i=0
    for R in rotation_matrices:

        rotated_points = cart_to_sph(sph_to_cart(points)@(R.T))
        # Compute the design matrix for the rotated points
        design_matrix, keys = eval_design_matrix_vectorized_3D(rotated_points, degree)
        # if np.min(np.linalg.svd(design_matrix)[1]) < 1e-5:
        #     print(R)
        #     print(R@R.T)
        #     print(np.linalg.det(R))
        #     sys.exit()
        # Append the design matrix for this rotation
        augmented_matrices.append(w[i]*design_matrix)
        i+=1
    return augmented_matrices

def weighted_least_squares(augmented_matrices, weights, y, tol):

    # Validate inputs
    if len(augmented_matrices) != len(weights):
        raise ValueError("Number of matrices and weights must be the same.")
    
    # Initialize weighted A and y
    y_weighted = []
    # Construct weighted matrices
    for w in weights:
        y_weighted.append(w * np.array(y))  # Scale the target vector

    # Stack the weighted contributions
    A_weighted = np.vstack(augmented_matrices)
    y_weighted = np.hstack(y_weighted)
    

    
    # Solve the least squares problem: A_weighted.T @ A_weighted @ beta = A_weighted.T @ y_weighted
    beta = np.linalg.lstsq(A_weighted, y_weighted, rcond=tol)[0]
    return beta

def clipped_values(original, N):
    """
    Return a copy of `original` where values are set to 0 if the key
    tuple violates either constraint:
      • any |entry| > N
      • |sum(entries)| > N
    """
    return {
        key: (
            0
            if any(abs(x) > N for x in key) or abs(sum(key)) > N
            else value
        )
        for key, value in original.items()
    }

def sample_SO3_via_QR(N):
    out = np.zeros((N,3,3))
    out[0, :] = np.eye(3)
    if N==1:
        return out
    for i in range(1,N):
        
        A = special_ortho_group.rvs(3)
        # Q, R = np.linalg.qr(A)               
        # if np.linalg.det(Q) < 0:             
        #     Q[:, 0] *= -1
        out[i, :] = A
    return out

def psym(points, coeffs, degree):
    """
    Approximates sym(f) through a order changeable quadrature rule
    """
    cart_points = sph_to_cart(points)
    filename = f"C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad_SO3//N{9}.dat"
    R_quad, w_quad = read_so3_quadrature_file(filename)
    out = np.zeros(len(points), dtype='complex128')
    i=0
    w_quad = np.array(w_quad)
    for R in R_quad:
        new_points = cart_to_sph(cart_points@(R.T))
        new_DM = eval_design_matrix_vectorized_3D(new_points, degree)[0]
        # print(new_DM.shape)
        # print(coeffs.shape)
        # print(w_quad[i])
        out += w_quad[i] * (new_DM @ coeffs)
        i+=1
    return out
    
def psym_invar(points, coeffs, degree):
    """
    Approximates sym(f) through a order changeable quadrature rule
    """
    cart_points = sph_to_cart(points)
    filename = f"/Users/hk/RAMathUBC/Quad_SO3/N{11}.dat"
    R_quad, w_quad = read_so3_quadrature_file(filename)
    out = np.zeros(len(points), dtype='complex128')
    i=0
    w_quad = np.array(w_quad)
    for R in R_quad:
        new_points = cart_to_sph(cart_points@(R.T))
        new_DM = eval_invar_design_matrix_3D(new_points, degree)[0]
        # print(new_DM.shape)
        # print(coeffs.shape)
        # print(w_quad[i])
        out += w_quad[i] * (new_DM @ coeffs)
        i+=1
    # I = np.eye(3)
    # contains_identity = any(np.array_equal(a, I) for a in R_quad)
    # if contains_identity:
    return out

def euler_from_matrix(mat):
    """
    Convert a 3x3 rotation matrix into Euler angles (ZYZ convention).
    
    Parameters:
    mat (array): A 3x3 rotation matrix.

    Returns:
    tuple: Euler angles (alpha, beta, gamma).
    """

    
    # return alpha, beta, gamma
    if np.array_equal(mat, np.eye(3)):
        return (0,0,0)
    r = sst.Rotation.from_matrix(mat)
    return r.as_euler('zyz')

def small_d_matrix(l, m_prime, m, beta):
    """
    Calculate the Wigner small d-matrix element d^l_{m', m}(\beta).

    Parameters:
    l (int): The total angular momentum.
    m_prime (int): The m' index.
    m (int): The m index.
    beta (float): The angle.

    Returns:
    float: The Wigner small d-matrix element d^l_{m', m}(\beta).
    """
    M = max(abs(m_prime), abs(m))
    N = min(abs(m_prime), abs(m))
    # Compute the d-matrix element
    prefactor = sqrt(factorial(l + M) * factorial(l - M) / (factorial(l + N) * factorial(l - N)))
    
    # The sine and cosine terms
    sine_term = sin(beta / 2) ** (abs(m_prime - m)) * cos(beta / 2) ** abs(m + m_prime)*eval_jacobi(l-M,abs(m-m_prime),abs(m+m_prime), cos(beta))
    
    return prefactor * sine_term * (-1)**((m-m_prime-abs(m-m_prime))/2)

def wigner_D_l(l, alpha, beta, gamma):
    """
    Generate the Wigner D^l matrix for given l and Euler angles (alpha, beta, gamma).

    Parameters:
    l (int): Degree of the Wigner matrix.
    alpha (float): First Euler angle.
    beta (float): Second Euler angle.
    gamma (float): Third Euler angle.

    Returns:
    numpy.ndarray: The Wigner D^l matrix of shape (2l+1, 2l+1).
    """
    size = 2 * l + 1
    D_matrix = np.zeros((size, size), dtype=complex)

    for m_prime in range(-l, l + 1):
        for m in range(-l, l + 1):
            d_l_mpm = small_d_matrix(l, m_prime, m, beta)  # Wigner small d-matrix element
            D_matrix[m_prime + l, m + l] = (
                np.exp(-1j * m_prime * alpha) * d_l_mpm * np.exp(-1j * m * gamma)
            )

    return D_matrix

    
def coupling_coeffs(coeffs):
    """
    Use generate_basis_keys_vectorized
    """
    m_prime = coeffs[::, 3::]
    tilde_C = np.zeros((coeffs.shape[0], m_prime.shape[0]), dtype='complex128')
    filename = f"/Users/hk/RAMathUBC/Quad_SO3/N{6}.dat"
    R_quad, w_quad = read_so3_quadrature_file(filename)
    EA_quad = np.zeros((len(R_quad),3))
    for i,R in enumerate(R_quad):
        EA_quad[i] = euler_from_matrix(np.array(R))
    for l,seq in tqdm(enumerate(coeffs)):
        for m, mp in enumerate(m_prime):
            if -seq[0]<=mp[0]<=seq[0] and -seq[1]<=mp[1]<=seq[1] and -seq[2]<=mp[2]<=seq[2]:
                coeff = 0
                for i, R in enumerate(EA_quad):
                    D_l1, D_l2, D_l3 = wigner_D_l(seq[0],R[0], R[1], R[2])[mp[0], seq[3]], wigner_D_l(seq[1], R[0], R[1], R[2])[mp[1],seq[4]], wigner_D_l(seq[2],R[0], R[1], R[2])[mp[2],seq[5]]

                    coeff += w_quad[i] * D_l1*D_l2*D_l3
                tilde_C[l,m] = coeff
    U,S,V = np.linalg.svd(tilde_C, full_matrices=False)
    print(S)
    r = np.sum(S>1e-3)
    # S[S<1e-3]=0.0
    print(r)
    
    return np.dot((U[:,:r]*S[:r]),V[:r,:])
    # return np.diag(S[:r]) @ V[:r, :]
    
def A(point,L ,mp):
    return np.array([sph_harm_y(L[0], seqm[0], point[::,0,1], point[::,0,0])*sph_harm_y(L[1], seqm[1], point[::,1,1], point[::,1,0])*sph_harm_y(L[2], seqm[2], point[::,2,1], point[::,2,0]) for seqm in mp])



def coupling_coeffs_test(seq, m_prime):
    """
    Use generate_basis_keys_vectorized
    """

    tilde_C = np.zeros(m_prime.shape[0], dtype='complex128')
    filename = f"/Users/hk/RAMathUBC/Quad_SO3/N{7}.dat"
    R_quad, w_quad = read_so3_quadrature_file(filename)
    EA_quad = np.zeros((len(R_quad),3))
    for i,R in enumerate(R_quad):
        EA_quad[i] = euler_from_matrix(np.array(R))
    
    for m, mp in enumerate(m_prime):
        coeff = 0
        for i, R in enumerate(EA_quad):
            D_l1, D_l2, D_l3 = wigner_D_l(seq[0],R[0], R[1], R[2]), wigner_D_l(seq[1], R[0], R[1], R[2]), wigner_D_l(seq[2],R[0], R[1], R[2])
            # D_l1, D_l2, D_l3 = wigner_D_l(seq[0],R[0], R[1], R[2])[seq[3], mp[0]], wigner_D_l(seq[1], R[0], R[1], R[2])[seq[4], mp[1]], wigner_D_l(seq[2],R[0], R[1], R[2])[seq[5], mp[2]]
            try:
                D_l1[seq[0]+mp[0],seq[0]+seq[3]]
            except:
                print(D_l1)

                print([seq[0]+mp[0],seq[0]+seq[3]])
                print('l1')
                print(seq)
                print(mp)
                sys.exit()
            try:
                D_l2[seq[1]+mp[1],seq[1]+seq[4]]
            except:
                print(D_l2)
                print([seq[1]+mp[1],seq[1]+seq[4]])
                print('l2')
                sys.exit()
            try:
                D_l3[seq[2]+mp[2],seq[2]+seq[5]]
            except:
                print(D_l3)
                print([seq[2]+mp[2],seq[2]+seq[5]])
                print('l3')
                sys.exit()
            coeff += w_quad[i] * D_l1[seq[0]+mp[0],seq[0]+seq[3]]*D_l2[seq[1]+mp[1],seq[1]+seq[4]]*D_l3[seq[2]+mp[2],seq[2]+seq[5]]
            # coeff += w_quad[i] * D_l1[seq[0]+seq[3], seq[0]+mp[0]]*D_l2[seq[1]+seq[4],seq[1]+mp[1]]*D_l3[seq[2]+seq[5],seq[2]+mp[2]]

        tilde_C[m] = coeff

    return tilde_C


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


# point = sample_spheres_uniform(10, 3)

# R = sample_SO3_via_QR(2)[1]

# test = gen_invar_basis_DM(2,point)
# sym_test = gen_invar_basis_DM(2, cart_to_sph(R@sph_to_cart(point)))

# print(test-sym_test)

# Test Symmetrisation

if __name__ != '__main__':
    point = sample_spheres_uniform(10, 3)
    test_degree = 5
    coeffs = generate_basis_keys_vectorized_3D(test_degree)
    filename = f"/Users/hk/RAMathUBC/Quad_SO3/N{3}.dat"
    R_quad, w_quad = read_so3_quadrature_file(filename)
    for l,seq in enumerate(np.unique(coeffs[:,:3],axis=0)):
        m_prime = []
        for m1 in range(-seq[0],seq[0]+1):
            for m2 in range(-seq[1],seq[1]+1):
                for m3 in range(-seq[2],seq[2]+1):
                    m_prime.append([m1,m2,m3])
        m_prime = np.array(m_prime)
        for m in m_prime:
            C = coupling_coeffs_test(np.concatenate((seq,m)), m_prime)
            test = A(point, seq, m_prime)
            B = C@test

            sym_test = np.zeros_like(B)
            for i, R in enumerate(R_quad):
                rotated =  A(cart_to_sph(sph_to_cart(point)@(R.T)),seq, m_prime)
                sym_test += w_quad[i]* C@rotated
            print(np.linalg.norm(B-sym_test))
                # print(B)
                # print(sym_test)
                # # print(np.linalg.norm(B-sym_test))
                # print(seq)
                # print(m)
                # print(C)
                # sys.exit()
             
    



# Test invariance

if __name__ != '__main__':
    test_points = sample_spheres_uniform(30, 3)
    target_degree = 5
    coeffs_target = gen_invar_coeffs_vectorized_3D(target_degree)
    target =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
    psym_target = psym_invar(test_points, np.array(list(coeffs_target.values())), target_degree)
    print(np.linalg.norm(psym_target-target))






# 3 fitting error

if __name__!= '__main__':
    degrees = [2,3,4,5,6,7]
    target_degree = 11
    # test_points = sample_spheres_uniform(2500, 3)
    # np.save(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC/test2500.npy', test_points)
    test_points = np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC/test2500.npy')
    # coeffs_target =  gen_invar_coeffs_vectorized_3D(target_degree)
    # np.save('/Users/hk/RAMathUBC/func3dkeys.npy', np.array(list(coeffs_target.keys())))
    # np.save('/Users/hk/RAMathUBC/func3dvalues.npy', np.array(list(coeffs_target.values())))
    keys = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//func3dkeys.npy')
    keys = [tuple(key) for key in keys]
    values = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//func3dvalues.npy')
    coeffs_target = dict(zip(keys, values))

    # target  =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
    # np.save(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC/target2500.npy', target)
    target = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//target2500.npy')
    trunc_error = []
    invar_error = []
    # proj_error = []
    N = 10000
    # points = sphere_2_const_pert(N, 3)
    # sample = evaluate_sum_of_invar_basis_3D(points, coeffs_target, target_degree)
    points = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//data3dD5samplepoints.npy')
    sample = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC/data3dD5samplevalues.npy')
    
    # np.save('/Users/hk/RAMathUBC/data3dD2testpoins.npy', test_points)
    # np.save('/Users/hk/RAMathUBC/data3dD1testvalues.npy', target)
    # np.save('/Users/hk/RAMathUBC/data3dD5samplepoints.npy', points)
    # np.save('/Users/hk/RAMathUBC/data3dD5samplevalues.npy', sample)
    Vol = (4/3*np.pi)**3
    
    tols = [1e-5]# D5[1e-5]# D3 [10**(-3.4)]
    for tol in tols:
        full_error = []
        trunc_error = []
        invar_error = []
        proj_error = []
        fig = plt.figure()
        plt.yscale('log')
        # plt.title(f'Approximating degree 11')
        plt.xlabel(r"Degree of Approximation", fontsize=17)
        # plt.ylabel(r"L2 Error", fontsize=17)
        plt.grid(True)
        plt.ylim(1e-7, 1e-3)
        plt.yticks([1e-3,1e-5,1e-7], fontsize=17)
        plt.xticks([2,4,6], fontsize=17)

        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        
        for degree in tqdm(degrees):
            DM_target = eval_design_matrix_vectorized_3D(test_points, degree)[0]
            # H = eval_proj_matrix(degree)
            coeffs_invar = fit_invar_least_squares_3D(points, sample, degree)
            val_invar = evaluate_invar_least_squares_3D(test_points ,coeffs_invar, degree)
            invar_error.append(np.linalg.norm(val_invar-target))
            DM, keys_full = eval_design_matrix_vectorized_3D(points, degree)
            # U,R,V = np.linalg.svd(DM)
            # r = R[R>tol]
            # rinv = np.diag(np.linalg.pinv(np.diag(r)))
            # sigma = np.zeros((V.shape[0], U.shape[0]))
            # np.fill_diagonal(sigma, rinv)
            # DM_r = V.T @ sigma @U.T
            # coeffs_full = DM_r @ sample
            coeffs_full = fit_least_squares_3D(points, sample, degree, tol)
            val_full = evaluate_full_least_squares_3D(test_points, coeffs_full, degree)
            val_proj = psym(test_points, np.array(list(coeffs_full.values())), degree)
            proj_error.append(np.linalg.norm(val_proj-target))
            # proj_full = H @ DM_target @ coeffs_full
            # proj_error.append(np.linalg.norm(proj_full-target))
            full_error.append(np.linalg.norm(val_full-target))
            trunc_coeffs = clipped_values(coeffs_target, degree)
            val_trunc = evaluate_sum_of_invar_basis_3D(test_points, trunc_coeffs, degree)
            trunc_error.append(np.linalg.norm(val_trunc-target))
        plt.plot(degrees, 1/(np.sqrt(2500)*Vol)*np.array(full_error), marker='o', color = 'tab:blue', label = f'LSQ: V ')
        plt.plot(degrees, 1/(np.sqrt(2500)*Vol)*np.array(trunc_error), marker='o', color='tab:orange', label = f'L2 Proj B')
        plt.plot(degrees, 1/(np.sqrt(2500)*Vol)*np.array(invar_error), marker='o', color = 'tab:green', label = f'LSQ: B')
        plt.plot(degrees, 1/(np.sqrt(2500)*Vol)*np.array(proj_error), marker='o', color = 'tab:red', label = f'LSQ:V + sym')

        # plt.legend(loc='upper right', fontsize=17)

        plt.show()
    

if __name__!= '__main__':
    degree = 4
    target_degree = 11
    test_points = np.load('/Users/hk/RAMathUBC/data3dD1testpoints.npy')
    # coeffs_target =  gen_invar_coeffs_vectorized_3D(target_degree)
    # np.save('/Users/hk/RAMathUBC/func3dkeys.npy', np.array(list(coeffs_target.keys())))
    # np.save('/Users/hk/RAMathUBC/func3dvalues.npy', np.array(list(coeffs_target.values())))
    keys = np.load('/Users/hk/RAMathUBC/func3dkeys.npy')
    keys = [tuple(key) for key in keys]
    values = np.load('/Users/hk/RAMathUBC/func3dvalues.npy')
    coeffs_target = dict(zip(keys, values))

    # target  =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
    target = np.load('/Users/hk/RAMathUBC/data3dD1testvalues.npy')
    trunc_error = []
    invar_error = []
    # proj_error = []
    N = 8000
    # points = sphere_1_const_pert(N, 3)
    # sample = evaluate_sum_of_invar_basis_3D(points, coeffs_target, target_degree)
    points = np.load('/Users/hk/RAMathUBC/data3dD5samplepoints.npy')
    sample = np.load('/Users/hk/RAMathUBC/data3dD5samplevalues.npy')

    tols = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
    full_error = []
    for tol in tqdm(tols):


        # plt.ylim(1e-5, 1e-1)
        # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        DM_target = eval_design_matrix_vectorized_3D(test_points, degree)[0]
        DM, keys_full = eval_design_matrix_vectorized_3D(points, degree)
        coeffs_full = fit_least_squares_3D(points, sample, degree, tol)
        val_full = evaluate_full_least_squares_3D(test_points, coeffs_full, degree)

        full_error.append(np.linalg.norm(val_full-target))
        # trunc_coeffs = clipped_values(coeffs_target, degree)
        # val_trunc = evaluate_sum_of_invar_basis_3D(test_points, trunc_coeffs, degree)
        # trunc_error.append(np.linalg.norm(val_trunc-target))
        
    fig = plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f'Approximating degree 11, D5, L=4')
    plt.xlabel("tol", )
    plt.ylabel("L2")
    plt.grid(True)
        
    plt.plot(tols, 1/np.sqrt(20)*np.array(full_error), marker='o', color = 'green', label = f'LSQ V')
    plt.plot(degrees, 1/np.sqrt(20)*np.array(trunc_error), marker='o', color='orange', label = f'L2 proj B')
    plt.plot(degrees, 1/np.sqrt(20)*np.array(invar_error), marker='o', color = 'blue', label = f'LSQ B')
    plt.plot(degrees, 1/np.sqrt(20)*np.array(proj_error), marker='o', color = 'red', label = f'LSQ V + sym')

    plt.legend(loc='upper right')

    plt.show()









# Quad Augmented data

if __name__ != '__main__':
    fig = plt.figure()
    degrees = [2,3,4,5,6] 
    target_degree = 8
    test_points = sample_spheres_uniform(200, 3)
    coeffs_target =  gen_invar_coeffs_vectorized_3D(target_degree)
    target =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
    Quad_error = []
    
    N = 800
    points = sample_spheres_uniform(N, 3)
    sample = evaluate_sum_of_invar_basis_3D(points, coeffs_target, target_degree)
    for N in tqdm(degrees):
        DM,keys = eval_design_matrix_vectorized_3D(test_points, N)
        filename = f'/Users/hk/RAMathUBC/Quad_SO3/N{N}.dat'
        R_quad, w_quad = read_so3_quadrature_file(filename)
        w_quad = np.sqrt(np.array(w_quad))
        Aug_DM_Quad = quad_augmented_design_matrix(points, N, R_quad, w_quad)
        coeff_quad = weighted_least_squares(Aug_DM_Quad, w_quad, sample)
        pred_quad = DM@ coeff_quad
        Quad_error.append(np.linalg.norm(target-pred_quad))
    plt.plot(degrees, 1/np.sqrt(20)*np.array(Quad_error), marker='o', label = 'Truncation Error')
    plt.yscale('log')
    plt.title('LSQ with Quad augmentation')
    plt.xlabel("Degree of LSQ and Quad", )
    plt.ylabel("L2 difference")
    plt.legend()
    plt.grid(True)
    plt.legend()




#Check LSQ

if __name__ != "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Full basis LSQ")
    plt.xlabel("Nb of data points")
    plt.ylabel("L2 norm")
    degree = 4
    target_degree = 6
    test_points = sample_spheres_uniform(20, 3)
    coeffs_target =  gen_invar_coeffs_vectorized_3D(target_degree)
    target =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
    full_error = []
    full_psym = []
    DM,keys = eval_design_matrix_vectorized_3D(test_points, degree)

    for N in tqdm([i for i in range(200,800,50)]):
        points = sample_spheres_uniform(N, 3)
        sample = evaluate_sum_of_invar_basis_3D(points, coeffs_target, target_degree)
        quad_psym = []
        DM_sample, keys_full = eval_design_matrix_vectorized_3D(points, degree)
        coeffs_full = np.linalg.lstsq(DM_sample, sample)[0] 
        val_full = DM @ coeffs_full
        full_error.append(np.linalg.norm(val_full-target))

        sym = psym(test_points, coeffs_full, degree)
        full_psym.append(np.linalg.norm(val_full - sym))
        
    plt.plot([i for i in range(200,800,50)], full_error, marker = "o", label = f'Quad : {degree} f-sym f')
    plt.plot([i for i in range(200,800,50)], full_psym, marker = "o", label = f'MC : {degree} f-sym f')
        # plt.plot([k for k in range(1,6,1)], quad_error, marker = "o", label = f'Quad : {degree} Approx Error')
        # plt.plot([k for k in range(1,6,1)], MC_error, marker = "o", label = f'MC : {degree} Approx Error')

    # plt.plot(degrees, np.exp(-0.40832958*np.array(degrees)), label=f'Exp fit')
    plt.legend()
    plt.grid()
    plt.show()

# Psym Quad

if __name__!= "__main__":

    degrees = [4,5,6]
    target_degree = 11
    #test_points = sample_spheres_uniform(200, 3)
    keys = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//func3dkeys.npy')
    keys = [tuple(key) for key in keys]
    values = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//func3dvalues.npy')
    coeffs_target = dict(zip(keys, values))
    colours = ['tab:blue', 'tab:orange', 'tab:green']
    # target  =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
    # np.save('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//target.npy', target)
    # np.save('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//test_points.npy', test_points)
    target = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//target.npy')
    test_points= np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//test_points.npy')

    nb_points = 800
    Vol = (4/3*np.pi)**3
    points = sample_spheres_uniform(nb_points, 3)
    sample = evaluate_sum_of_invar_basis_3D(points, coeffs_target, target_degree)
    np.save('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//sampleD1.npy', sample)
    np.save('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//pointsD1.npy', points)
    # sample = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//sampleD4.npy')
    # points = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//pointsD4.npy')
    tol = None # D4/5[1e-5]# D3 [10**(-3.4)]
    plt.figure()
    # plt.title("Symmetrisation methods (2,pert)")
    plt.xlabel("Degree of Quadrature", fontsize=17)
    plt.ylabel("||f-sym f||", fontsize=17)
    plt.yscale("log")
    plt.ylim(1e-18, 1e0)
    plt.yticks([1e0, 1e-6, 1e-12, 1e-18], fontsize=17)
    # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.xticks([2,4,6], fontsize=17)
    for a,degree in enumerate(degrees):
        # Quad_psym= np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//PysmD4deg{degree}.npy')
        # Quad_error= np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//L2D4deg{degree}.npy')
        MC_psym = []
        MC_error = []
        Quad_psym = []
        Quad_error = []
        coeffs_invar = fit_invar_least_squares_3D(points, sample, degree)
     
        DM,keys = eval_design_matrix_vectorized_3D(test_points, degree)
        for N in tqdm([i for i in range(1,8)]):
            filename = f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad_SO3//N{N}.dat'
            R_quad, w_quad = read_so3_quadrature_file(filename)
            w_quad = np.sqrt(np.array(w_quad))
            # R_MC = sample_SO3_via_QR(len(R_quad))
            # w_MC =np.ones(len(R_quad))

            # Aug_DM_MC = quad_augmented_design_matrix(points, degree, R_MC, w_MC)
            # coeff_MC = weighted_least_squares(Aug_DM_MC, w_MC, sample)
            # pred_MC = DM @ coeff_MC
            
            # sym_pred_MC = psym(test_points, coeff_MC, degree)
            # MC_error.append(np.linalg.norm(target-pred_MC))
            # MC_psym.append(np.linalg.norm(sym_pred_MC-pred_MC))
            
            Aug_DM_Quad = quad_augmented_design_matrix(points, degree, R_quad, w_quad)
            coeff_quad = weighted_least_squares(Aug_DM_Quad, w_quad, sample, tol)
            pred_quad = DM@ coeff_quad
            
            sym_pred_quad = psym(test_points, coeff_quad, degree)
            Quad_error.append(np.linalg.norm(target-pred_quad))
            Quad_psym.append(np.linalg.norm(sym_pred_quad-pred_quad))
        Quad_error = 1/(np.sqrt(200)*Vol)* np.array(Quad_error)
        Quad_psym =1/(np.sqrt(200)*Vol)*  np.array(Quad_psym)
            
        # plt.plot([i for i in range(1,8)], MC_psym, marker = "o", label = f'Degree {degree}, MC')
        plt.plot([i for i in range(1,8)], Quad_psym, marker = "o", color = colours[a], label = f'Deg : {degree}')
        plt.plot([i for i in range(1,8)], Quad_error, marker = "o", color = colours[a], alpha = 0.2)

        np.save(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//PysmD1deg{degree}.npy', Quad_psym)
        np.save(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//L2D1deg{degree}.npy', Quad_error)

        # plt.plot([k for k in range(1,6,1)], quad_error, marker = "o", label = f'Quad : {degree} Approx Error')
        # plt.plot([k for k in range(1,6,1)], MC_error, marker = "o", label = f'MC : {degree} Approx Error')

    plt.legend(loc = 'upper right', fontsize=17)
    plt.grid()
    plt.show()


# check L2 equiv to Linf
if __name__!= "__main__":
    degrees = [2,3,4,5,6]
    target_degree = 11

    plt.figure()
    plt.ylabel(r'L2/Linf')
    plt.xlabel(r'degree')
    psyms = np.zeros((10,5,2))
    for i in tqdm(range(10)):
        points = sample_spheres_uniform(100, 3)
        test_points = sample_spheres_uniform(100, 3)
        coeffs_target =  gen_invar_coeffs_vectorized_3D(target_degree)
        target =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
        for j,deg in enumerate(degrees):
            DM,keys = eval_design_matrix_vectorized_3D(test_points,deg)
            DM_lstsq, _ = eval_design_matrix_vectorized_3D(points,deg)

            coeff_test = np.linalg.lstsq(DM_lstsq, target)[0]
            pred_test = DM@ coeff_test
            sym_pred_test = psym(test_points, coeff_test, deg)

            l2 = np.linalg.norm(pred_test-sym_pred_test)
            linf = np.max(np.abs(pred_test-sym_pred_test))
            psyms[i,j,0]=l2
            psyms[i,j,1]=linf
            
            

    mean_psyml2 = np.mean(psyms[:, :, 0], axis = 0)
    stvd_psyml2 = np.std(psyms[:, :, 0], axis = 0)
    mean_psymlinf = np.mean(psyms[:, :, 1], axis = 0)
    stvd_psymlinf = np.std(psyms[:, :, 1], axis = 0)
    ratio = mean_psyml2/mean_psymlinf
    # plt.plot(rots, mean_psym, marker="", label=f'Mean : {degree}')
    plt.errorbar(degrees, ratio, np.abs(ratio * stvd_psyml2/mean_psyml2 - ratio * stvd_psymlinf/mean_psymlinf), marker = 'o', label=r'Mean +/- stvd')


    plt.legend()
    plt.grid()
    plt.show()











#MC Psym

if __name__ != "__main__":
    for dist in [1,2,3,4,5]:
        degrees = [4,5,6]
        # target_degree = 11
        # Vol = (4/3*np.pi)**2
        # keys = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//func3dkeys.npy')
        # keys = [tuple(key) for key in keys]
        # values = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//func3dvalues.npy')
        # coeffs_target = dict(zip(keys, values))
        rots = [1,2,4,8,16,32,64,128, 256]
        # if dist == 3:
        #     tol = 10**(-3.4)
        #     # test_points = sphere_1_const_pert(15, 3)
        # elif dist == 4:
        #     tol = 10**(-5)
        #     # test_points = sphere_2_constraint(15, 3)
        # elif dist == 5:
        #     tol = 10**(-5)
        #     # test_points = sphere_2_const_pert(15, 3)
        # elif dist == 2: 
        #     tol = 10 **(-3.4)
        #     # test_points = sphere_1_constraint(15, 3)
        # elif dist == 1: 
        #     tol = None
        #     # test_points = sample_spheres_uniform(15, 3)
        # # target  =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
        # # target = np.load('C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//data3dD1testvalues.npy')
        # test_points = np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//test_points.npy')
        # target = np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//target.npy')
        # TS = target.shape[0]
        # all_psyms = np.zeros((10,len(degrees),len(rots)))
        colours = ['tab:blue', 'tab:orange', 'tab:green']
        # for i in tqdm(range(10)):
        #     # target =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
        #     trunc_error = []
        #     full_error = []
        #     invar_error = []
        #     nb_points = 800
            
        #     # if dist == 1:
        #     #     points = sample_spheres_uniform(nb_points, 3)
                
        #     # elif dist == 2:
        #     #     points = sphere_1_constraint(nb_points, 3)
            
        #     # elif dist == 3:
        #     #     points = sphere_1_const_pert(nb_points, 3)
            
        #     # elif dist == 4:
        #     #     points = sphere_2_constraint(nb_points, 3)
                
        #     # elif dist == 5:
        #     #     points = sphere_2_const_pert(nb_points, 3)

        #     # sample = evaluate_sum_of_invar_basis_3D(points, coeffs_target, target_degree)


        #     sample = np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//sampleD{dist}.npy')
        #     points = np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//pointsD{dist}.npy')
            
        #     # points = np.load(f'C://Users//zziyu//OneDrive - University of Toronto//Desktop//RAMathUBC//RAMathUBC//MC3d//dataMC3dD1points{i}v3.npy')
        #     # sample = np.load(f'C://Users//zziyu//OneDrive - University of Toronto//Desktop//RAMathUBC//RAMathUBC//MC3d//dataMC3dD1sample{i}v3.npy')
    
        #     for a,degree in enumerate(degrees):
        #         MC_psym = []
        #         MC_error = []
        #         for N in rots:
                
        #             # rotated_test_points= np.zeros((N*TS, 3, 2))
        #             # rotated_target = np.zeros(N*TS, dtype="complex")
        #             R_MC = sample_SO3_via_QR(N)
        #             w_MC =np.sqrt(np.ones(N)/N)
        #             # for r, R in enumerate(R_MC):
        #             #     rotated_test_points[r*TS:(r+1)*TS, :] = cart_to_sph(sph_to_cart(test_points)@(R.T))
        #             #     rotated_target[r*TS:(r+1)*TS] = target
        #             # DM,keys = eval_design_matrix_vectorized_3D(rotated_test_points, degree)
                    
        #             DM,keys = eval_design_matrix_vectorized_3D(test_points, degree)
        #             Aug_DM_MC = quad_augmented_design_matrix(points, degree, R_MC, w_MC)
        #             coeff_MC = weighted_least_squares(Aug_DM_MC, w_MC, sample, tol)
        #             pred_MC = DM @ coeff_MC
                    
                    
        #             sym_pred_MC = psym(test_points, coeff_MC, degree)
        #             MC_error.append(1/(np.sqrt(200)*Vol)*np.linalg.norm(target-pred_MC))
        #             MC_psym.append(1/(np.sqrt(200)*Vol)*np.linalg.norm(sym_pred_MC-pred_MC))
        #             if i == 0:
        #                 np.save(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//MCL2D{dist}deg{degree}.npy', MC_error)
        #         all_psyms[i,a] = MC_psym
        
        
        # np.save(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//datapsym3dMC800all_psymsD{dist}.npy', all_psyms)
    
        # plt.figure()
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.ylim(1e-5,1e-1)
        # for i, degree in enumerate(degrees):
        #     plt.plot(rots, all_psyms[0,i,:], label=f'{degree}')
        # plt.plot(np.logspace(0,2.5,100), 0.002*np.logspace(0 ,2.5,100)**(-0.5), linestyle = "--",  label = f'T^-1/2')
        # plt.legend()
        # plt.show()
        # rots = [1,2,4,8,16,32,64,128, 256, 512]
        dist=1
        all_psyms = np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//datapsym3dMC800all_psymsD{dist}.npy')

        plt.figure()
            # plt.title("Symmetrisation methods (2,pert)")
        rots = [1,2,4,8,16,32,64,128, 256]

        plt.xlabel("Nb of rotations", fontsize=17)
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(1e-8, 1e-3)
        if dist == 1 or dist ==4:
            plt.ylabel("||f-sym f||", fontsize=17)

            plt.yticks([1e-7, 1e-5, 1e-3], fontsize=17)
        plt.xticks([1e1,1e2], fontsize=17)
        if dist !=1 and dist !=4: 
            plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        plt.plot(np.logspace(0,2.5,300), 0.000005*np.logspace(0 ,2.5,300)**(-0.5), linestyle = "--",  label = f'T^-1/2')
  
        for a, degree in enumerate(degrees):
            # l2 = np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//MCL2D{dist}deg{degree}.npy')
            Quad_psym= np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//PysmD{dist}deg{degree}.npy')
            Quad_error= np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//Quad3d//L2D{dist}deg{degree}.npy')
            mean_psym = np.mean(all_psyms[:,a,:], axis = 0)
            stvd_psym = np.std(all_psyms[:,a,:], axis =0)
            # plt.plot(rots, mean_psym)
            plt.plot(rots, mean_psym, marker = 'o', color=colours[a], alpha = 0.2)
            plt.plot([4,11,23,43,60,116, 168], Quad_psym, marker = 'o', color=colours[a], alpha = 1, label=f'Deg : {degree}')

            #plt.errorbar(rots, mean_psym, yerr = [np.zeros_like(stvd_psym), stvd_psym], linestyle=None, lolims=True, color=colours[a], ecolor='black', label=f'Mean+stdv : {degree}')
        # plt.legend(loc='upper right', fontsize=17)
        plt.grid()
        if dist ==1:
            plt.legend(loc = 'upper right', fontsize=17)
        plt.show()


# for dist in [1,2,3,4,5]:
#     all_psyms = np.load(f'C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//datapsym3dMCD{dist}v3_Augcheck.npy')
#     plt.figure()
#             # plt.title("Symmetrisation methods (2,pert)")
#     if dist == 1:
#         plt.xlabel("Nb of rotations")
#         # plt.ylabel("||f-sym f||")
#     plt.yscale("log")
#     plt.xscale("log")
#     plt.ylim(1e-5, 1e-1)
#     plt.plot(np.logspace(0,3,300), 0.0015*np.logspace(0 ,3,300)**(-0.5), linestyle = "--",  label = f'T^-1/2')
#     if dist > 1:
#         plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
#     for a, degree in enumerate(degrees):
#         mean_psym = np.mean(all_psyms[:,a,:], axis = 0)
#         stvd_psym = np.std(all_psyms[:,a,:], axis =0)
#             # plt.plot(rots, mean_psym)
#         plt.errorbar(rots, mean_psym, yerr = [np.zeros_like(stvd_psym), stvd_psym], linestyle=None, lolims=True, ecolor='black', label=f'Mean+stdv : {degree}')
#     plt.legend(prop={'size':12})
#     plt.grid()
#     plt.show()




if __name__ != "__main__":
    rots = [128, 256]
    degrees = [4,5,6]
    target_degree = 11
    test_points = np.load('C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//data3dD1testpoints.npy') 
    keys = np.load('C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//func3dkeys.npy')
    keys = [tuple(key) for key in keys]
    values = np.load('C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//func3dvalues.npy')
    coeffs_target = dict(zip(keys, values))
    target = np.load('C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//data3dD1testvalues.npy')
    tol = None
    all_psyms = np.zeros((10,len(degrees), len(rots)))
    for i in tqdm(range(10)):
        # target =  evaluate_sum_of_invar_basis_3D(test_points, coeffs_target, target_degree)
        trunc_error = []
        full_error = []
        invar_error = []
        nb_points = 1000
        

        # np.save(f'C://Users//zziyu//OneDrive - University of Toronto//Desktop//RAMathUBC//RAMathUBC//MC3d//dataMC3dD{dist}points{i}v3.npy', points)
        # np.save(f'C://Users//zziyu//OneDrive - University of Toronto//Desktop//RAMathUBC//RAMathUBC//MC3d//dataMC3dD{dist}sample{i}v3.npy', sample)
        
        points = np.load(f'C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//MC3d//dataMC3dD1points{i}v3.npy')
        sample = np.load(f'C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//MC3d//dataMC3dD1sample{i}v3.npy')

        for a,degree in enumerate(degrees):
            MC_psym = []
            MC_error = []
            DM,keys = eval_design_matrix_vectorized_3D(test_points, degree)
            for N in tqdm(rots):
    
    
                R_MC = sample_SO3_via_QR(N)
                w_MC =np.ones(N)/N
    
                Aug_DM_MC = quad_augmented_design_matrix(points, degree, R_MC, w_MC)
                coeff_MC = weighted_least_squares(Aug_DM_MC, w_MC, sample, tol)
                pred_MC = DM @ coeff_MC
    
                
                sym_pred_MC = psym(test_points, coeff_MC, degree)
                MC_error.append(np.linalg.norm(target-pred_MC))
                MC_psym.append(np.linalg.norm(sym_pred_MC-pred_MC))
            all_psyms[i,a] = MC_psym
    
    
    # np.save('C://Users//zziyu//OneDrive - University of Toronto//Desktop//RAMathUBC//RAMathUBC//datapsym3dMCD{dist}.npy', all_psyms)

    # plt.figure()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(1e-5,1e-1)
    # for i, degree in enumerate(degrees):
    #     plt.plot(rots, all_psyms[0,i,:], label=f'{degree}')
    # plt.plot(np.logspace(0,2.5,100), 0.002*np.logspace(0 ,2.5,100)**(-0.5), linestyle = "--",  label = f'T^-1/2')
    # plt.legend()
    # plt.show()

    
    # plt.figure()
    #     # plt.title("Symmetrisation methods (2,pert)")
    # plt.xlabel("Nb of rotations")
    # # plt.ylabel("||f-sym f||")
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.ylim(1e-5, 1e-1)
    # plt.plot(np.logspace(0,2,200), 0.0008*np.logspace(0 ,2,200)**(-0.5), linestyle = "--",  label = f'T^-1/2')

    # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # for a, degree in enumerate(degrees):
    #     mean_psym = np.mean(all_psyms[:,a,:], axis = 0)
    #     stvd_psym = np.std(all_psyms[:,a,:], axis =0)
    #     # plt.plot(rots, mean_psym)
    #     plt.errorbar(rots, mean_psym, yerr = [np.zeros_like(stvd_psym), stvd_psym], linestyle=None, lolims=True, ecolor='black', label=f'Mean+stdv : {degree}')
    # plt.legend(prop={'size':12})
    # plt.grid()
    # plt.show()





#  Verlet 3D

# def J(vs: np.ndarray) -> float:
    
    
    
    
#     return np.sqrt(vs[0,0]**2+vs[0,1]**2*np.sin(vs[0,0])**2)+np.sqrt(vs[1,0]**2+vs[1,1]**2*np.sin(vs[1,0])**2)+np.sqrt(vs[2,0]**2+vs[2,1]**2*np.sin(vs[2,0])**2)




# def wrap_angles_inplace(q, v):
#     """
#     In-place wrap for spherical angles.
#     q: (3,2) [theta, phi]; v: (3,2) [theta_dot, phi_dot]
#       - Keep phi in [0, 2π).
#       - Keep theta in [0, π] using (theta, phi) ~ (2π - theta, phi + π).
#         When we mirror theta -> 2π - theta, flip v_theta -> -v_theta.
#         Actually switch theta and phi
#     """
#     TWOPI = np.pi*2
#     # wrap phi to [0, 2π)
#     q[:, 0] = np.mod(q[:, 0], TWOPI)

#     # bring theta into [0, 2π)
#     q[:, 1] = np.mod(q[:, 1], TWOPI)

#     # for any theta > π, mirror to (2π - theta) and shift phi by π
#     mask = q[:, 1] > np.pi
#     if np.any(mask):
#         q[mask, 1] = TWOPI - q[mask, 1]      # theta -> 2π - theta
#         q[mask, 0] = q[mask, 0] + np.pi         # phi   -> phi + π
#         v[mask, 1] = -v[mask, 1]             # flip theta velocity due to mirroring

#     # re-wrap phi again (after potential +π)
#     q[:, 0] = np.mod(q[:, 0], TWOPI)
#     return q,v

# def eval_partial_sph_basis_3D(l1, l2, l3, m1, m2, m3, points, k):
#     if m1 + m2 + m3 != 0:
#         return np.zeros(points.shape[0], dtype=np.complex128)

#     N = points.shape[0]
#     result = np.zeros(shape=(2), dtype=np.complex128)

#     phi1, theta1 = points[0, 0], points[0, 1]
#     phi2, theta2 = points[1, 0], points[1, 1]
#     phi3, theta3 = points[2, 0], points[2, 1]

#     L_min = abs(l1 - l2)
#     L_max = l1 + l2

#     if k== 0:
#         for mu1 in range(-l1, l1+1):
#             for mu2 in range(-l2, l2+1):
#                 mu3 = -mu2 -mu1
#                 if -l3<= mu3<=l3 and L_min <= l3 <= L_max:
#                     L = l3
                    
    
#                     C_mm = float(clebsch_gordan(S(l1), S(l2), S(L), S(m1), S(m2), S(m1 + m2)))
#                     C_mu = float(clebsch_gordan(S(l1), S(l2), S(L), S(mu1), S(mu2),  S(mu1 + mu2)))
                
#                     C_mu3 = float(clebsch_gordan(S(L), S(l3), S(0), S(mu1 + mu2), S(mu3), S(0)))
#                     C_mm3 = float(clebsch_gordan(S(L), S(l3), S(0),S(m1 + m2), S(m3), S(0)))
                    
                       
#                     coeff = C_mm * C_mu * C_mm3 * C_mu3
    
#                     Y1 = sph_harm_y(l1, mu1, theta1, phi1, diff_n=1)[1]
#                     Y2 = sph_harm_y(l2, mu2, theta2, phi2)
#                     Y3 = sph_harm_y(l3, mu3, theta3, phi3)
#                     result += (coeff * Y2 * Y3) * Y1
#         return result

#     if k== 1:
#         for mu1 in range(-l1, l1+1):
#             for mu2 in range(-l2, l2+1):
#                 mu3 = -mu2 -mu1
#                 if -l3<= mu3<=l3 and L_min <= l3 <= L_max:
#                     L = l3
                    
    
#                     C_mm = float(clebsch_gordan(S(l1), S(l2), S(L), S(m1), S(m2), S(m1 + m2)))
#                     C_mu = float(clebsch_gordan(S(l1), S(l2), S(L), S(mu1), S(mu2),  S(mu1 + mu2)))
                
#                     C_mu3 = float(clebsch_gordan(S(L), S(l3), S(0), S(mu1 + mu2), S(mu3), S(0)))
#                     C_mm3 = float(clebsch_gordan(S(L), S(l3), S(0),S(m1 + m2), S(m3), S(0)))
                    
                       
#                     coeff = C_mm * C_mu * C_mm3 * C_mu3
    
#                     Y1 = sph_harm_y(l1, mu1, theta1, phi1)
#                     Y2 = sph_harm_y(l2, mu2, theta2, phi2, diff_n=1)[1]
#                     Y3 = sph_harm_y(l3, mu3, theta3, phi3)
#                     result += (coeff * Y2 * Y3) * Y1
#         return result
#     if k== 2:
#         for mu1 in range(-l1, l1+1):
#             for mu2 in range(-l2, l2+1):
#                 mu3 = -mu2 -mu1
#                 if -l3<= mu3<=l3 and L_min <= l3 <= L_max:
#                     L = l3
                    
    
#                     C_mm = float(clebsch_gordan(S(l1), S(l2), S(L), S(m1), S(m2), S(m1 + m2)))
#                     C_mu = float(clebsch_gordan(S(l1), S(l2), S(L), S(mu1), S(mu2),  S(mu1 + mu2)))
                
#                     C_mu3 = float(clebsch_gordan(S(L), S(l3), S(0), S(mu1 + mu2), S(mu3), S(0)))
#                     C_mm3 = float(clebsch_gordan(S(L), S(l3), S(0),S(m1 + m2), S(m3), S(0)))
                    
                       
#                     coeff = C_mm * C_mu * C_mm3 * C_mu3
    
#                     Y1 = sph_harm_y(l1, mu1, theta1, phi1)
#                     Y2 = sph_harm_y(l2, mu2, theta2, phi2)
#                     Y3 = sph_harm_y(l3, mu3, theta3, phi3, diff_n=1)[1]
#                     result += (coeff * Y2 * Y3) * Y1
#         return result

                



# def inv_sin_or_zero(x, tol=1e-12):
#     x = np.asarray(x, dtype=float)
#     s = np.sin(x)                       # if your x is in degrees: s = np.sin(np.deg2rad(x))
#     mask = np.isfinite(s) & (np.abs(s) > tol)
#     out = np.zeros_like(s)
#     np.divide(1.0, s, out=out, where=mask)
#     return out



# def evaluate_grad_of_invar_basis_3D(points, coeffs, degree, epsilon):
#     """
#     Evaluate the function f(points) = sum c_{l1l2l3m1m2m3} * B_{l1l2l3}^{m1m2m3}(points)

#     Parameters:
#         points: ndarray of shape (N, 3, 2)
#         coeffs: dict with keys (l1, l2, l3, m1, m2, m3) and real values
#         degree: int, upper bound on l1 + l2 + l3

#     Returns:
#         values: ndarray of shape (N,) with complex values
#     """
#     N = 2
#     result0 = np.zeros(N, dtype=np.complex128)
#     result1 = np.zeros(N, dtype=np.complex128)
#     result2 = np.zeros(N, dtype=np.complex128)

#     for (l1, l2, l3, m1, m2, m3), c_val in coeffs.items():
#         if l1 + l2 + l3 > degree:
#             continue
#         if m1 + m2 + m3 != 0:
#             continue
#         basis_val0 = eval_partial_sph_basis_3D(l1, l2, l3, m1, m2, m3, points, 0)
#         basis_val1 = eval_partial_sph_basis_3D(l1, l2, l3, m1, m2, m3, points, 1)
#         basis_val2 = eval_partial_sph_basis_3D(l1, l2, l3, m1, m2, m3, points, 2)

#         result0 += c_val * basis_val0
#         result1 += c_val * basis_val1
#         result2 += c_val * basis_val2
#     c1 = inv_sin_or_zero(points[0,1])
#     c2 = inv_sin_or_zero(points[1,1])
#     c3 = inv_sin_or_zero(points[2,1])

#     return np.array([[c1*(result0[1]+epsilon*np.cos(points[0,1])*np.sin(points[0,0])), (result0[0]+epsilon*np.sin(points[0,1])*np.cos(points[0,0]))], 
#                      [c2*(result1[1]+epsilon*np.cos(points[1,1])*np.sin(points[1,0])), (result1[0]+epsilon*np.sin(points[1,1])*np.cos(points[1,0]))],
#                      [c3*(result2[1]+epsilon*np.cos(points[2,1])*np.sin(points[2,0])), (result2[0]+epsilon*np.sin(points[2,1])*np.cos(points[2,0]))]])







# def verlet_integrate(points: np.ndarray,
#                      vs: np.ndarray,
#                      dt: float,
#                      n_steps: int,
#                      coeffs,
#                      epsilon):

#     points = points.astype(float).copy()
#     vs = vs.astype(float).copy()

#     J_series = np.empty(n_steps + 1)
#     J_series[0] = J(vs)

#     for n in tqdm(range(1, n_steps + 1)):
#         # a(t)  ---------------------------------------------------------------
#         a      = -evaluate_grad_of_invar_basis_3D(points, coeffs,6 ,epsilon).real
#         # θ(t+dt) -------------------------------------------------------------
#         points += vs * dt + 0.5 * a * dt**2

#         # a(t+dt) -------------------------------------------------------------
#         a_new  = -evaluate_grad_of_invar_basis_3D(points, coeffs, 6,epsilon).real

#         # ω(t+dt) -------------------------------------------------------------
#         vs += 0.5 * (a + a_new) * dt
#         points, vs = wrap_angles_inplace(points, vs)
#         # store K -------------------------------------------------------------
#         J_series[n] = J(vs)
        
#     return J_series



# def hitting_time(points: np.ndarray,
#                      vs: np.ndarray,
#                      dt: float,
#                      coeffs,
#                      epsilon,
#                      tol):

#     points = points.astype(float).copy()
#     vs = vs.astype(float).copy()

#     J_series = np.ones(10000)
#     J_series[0] = J(vs)
#     n=0
#     while np.abs(J_series[n]-J_series[0])<=tol and 1.0 in J_series:

#         n+=1

#         if n % 100 == 0:
#             print(n)

#         # a(t)  ---n------------------------------------------------------------
#         a      = -evaluate_grad_of_invar_basis_3D(points, coeffs,6 ,epsilon).real
#         # θ(t+dt) -------------------------------------------------------------
#         points += vs * dt + 0.5 * a * dt**2

#         # a(t+dt) -------------------------------------------------------------
#         a_new  = -evaluate_grad_of_invar_basis_3D(points, coeffs, 6,epsilon).real

#         # ω(t+dt) -------------------------------------------------------------
#         vs += 0.5 * (a + a_new) * dt
#         points, vs = wrap_angles_inplace(points, vs)
#         # store K -------------------------------------------------------------
#         J_series[n] = J(vs)
#     return n, J_series





