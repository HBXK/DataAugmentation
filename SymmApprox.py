#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:22:53 2024

@author: hk
"""

import numpy as np
from scipy.special import sph_harm
from scipy.linalg import expm, eigh
from itertools import product
import matplotlib.pyplot as plt
from sympy.physics.quantum.cg import CG
import quaternionic
import spherical
ell_max = 16  # Use the largest ℓ value you expect to need
wigner = spherical.Wigner(ell_max)


def points_sphere_gaussian(N):

    points = np.random.randn(N, 3) 
    norms = np.linalg.norm(points, axis=1, keepdims=True)  
    return points / norms 

def sample_concentrated_sphere(N, kappa=10):
    """
    Samples N points from a von Mises-Fisher distribution centered around the north pole.
    
    Parameters:
    - N (int): Number of points to sample.
    - kappa (float): Concentration parameter. Higher values concentrate points closer to the north pole.
    
    Returns:
    - points (np.ndarray): Array of shape (N, 3) with sampled points on the unit sphere.
    """
    # Step 1: Sample points from a von Mises-Fisher distribution around the north pole [0, 0, 1].
    # For large kappa, points concentrate near the north pole.
    
    # Generate random azimuthal angles uniformly
    phi = np.random.uniform(0, 2 * np.pi, N)
    
    # Sample polar angle with concentration kappa
    theta = np.arccos(1 - np.log(np.random.uniform(0, 1, N) * (np.exp(kappa) - np.exp(-kappa)) + np.exp(-kappa)) / kappa)
    
    # Step 2: Convert spherical to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Combine into an (N, 3) array
    points = np.stack([x, y, z], axis=1)
    return points

def gram_matrix_3d(points):

    points = np.asarray(points)
    if points.shape != (3, 3):
        raise ValueError("Input must have shape (3, 3), representing three points on the sphere.")
    
    gram_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            gram_matrix[i, j] = np.dot(points[i], points[j])

    return gram_matrix



def trace_exponential(G):

    eigenvalues = eigh(G, eigvals_only=True)  # Compute only the eigenvalues
    return np.sum(np.exp(eigenvalues))  # Sum the exponential of the eigenvalues


def f_3d_efficient(Gs):

    return np.array([trace_exponential(G) for G in Gs])



def cartesian_to_spherical(points):

    x, y, z = points.T
    r = 1 #np.linalg.norm(points, axis=1)  # Compute radius
    theta = np.arccos(z / r)  # Polar angle
    phi = np.arctan2(y, x)  # Azimuthal angle
    return np.column_stack((theta, phi))


# def generate_spherical_harmonic_terms(degree):

#     terms = []
#     for l1 in range(degree + 1):
#         for l2 in range(degree + 1 - l1):
#             l3 = degree - l1 - l2
#             for m1 in range(-l1, l1 + 1):
#                 for m2 in range(-l2, l2 + 1):
#                     for m3 in range(-l3, l3 + 1):
#                         terms.append(((l1, m1), (l2, m2), (l3, m3)))
#     return terms

def sph_harm_terms(degree):
    terms = []
    for l1 in range(degree+1):
        for l2 in range(degree+1):
            for l3 in range(np.abs(l1-l2), l1+l2+1):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        for m3 in range(-l3, l3 + 1):
                            terms.append(((l1, m1), (l2, m2), (l3, m3)))


def recursion_formula(l,mu,m):
    if len(m)>1:
        
        out = 0
        for L in range(np.abs(l[0]-l[1]),l[0]+l[1]+1):
            out+= CG(l[0],m[0],l[1],m[1],L, m[0]+m[1])*CG(l[0],mu[0],l[1],mu[1],L, mu[0]+mu[1])
            *recursion_formula(np.concatenate([L],L[2::]), np.concatenate([mu[0]+mu[1]],mu[2::]),np.concatenate([m[0]+m[1]],m[2::]))
    else:
        return wigner()
            

def invariant_powers(degree):
    




def spherical_harmonic_design_matrix_3points(points, degree):

    # Ensure the input is (N, 3, 3)
    points = np.asarray(points)
    if points.shape[1:] != (3, 3):
        raise ValueError("Input must have shape (N, 3, 3), representing N sets of 3 points.")

    # Convert each point to spherical angles
    angles = np.array([cartesian_to_spherical(p) for p in points])  # Shape: (N, 3, 2)

    # Extract theta and phi for each point
    theta1, phi1 = angles[:, 0, 0], angles[:, 0, 1]
    theta2, phi2 = angles[:, 1, 0], angles[:, 1, 1]
    theta3, phi3 = angles[:, 2, 0], angles[:, 2, 1]

    # Generate spherical harmonic terms
    terms = sph_harm_terms(degree)

    # Compute the design matrix
    design_matrix = []
    for (l1, m1), (l2, m2), (l3, m3) in terms:
        # Evaluate spherical harmonics for the respective points
        Y1 = sph_harm(m1, l1, phi1, theta1)
        Y2 = sph_harm(m2, l2, phi2, theta2)
        Y3 = sph_harm(m3, l3, phi3, theta3)

        # Triple product for all points
        triple_product = Y1 * Y2 * Y3

        # Add the column to the design matrix
        design_matrix.append(triple_product)

    # Stack columns into a matrix
    return np.column_stack(design_matrix)



def spherical_harmonics_design_matrix(configurations, d):
    harmonics = []
    
    for config in configurations:
        config = np.atleast_2d(config)
       
        if config.shape[1] != 3:
           raise ValueError("Each configuration must have shape (N_part, 3)")

       # Convert Cartesian coordinates to spherical angles
        theta = np.arccos(np.clip(config[:, 2], -1, 1))  # polar angle (z-axis projection)
        phi = np.arctan2(config[:, 1], config[:, 0])     # azimuthal angle (x-y plane)

        #theta = np.arccos(config[:,2])  # polar angle for each particle
        #phi = np.arctan2(config[:,1], config[:,0])  # azimuthal angle for each particle
        row = []
        for l in range(d + 1):
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, phi, theta)
                row.extend(Y_lm)  # Use the real part only ?
        harmonics.append(row)
    return np.array(harmonics)

def augmented_design_matrices(points, degree, rotation_matrices):

    augmented_matrices = []
    
    for R in rotation_matrices:
        # Apply rotation matrix R to all points
        rotated_points = np.array([R @ p.T for p in points])  # Shape (N, 3, 3)
        rotated_points = rotated_points.transpose(0, 2, 1)  # Reshape to (N, 3, 3)

        # Compute the design matrix for the rotated points
        design_matrix = spherical_harmonic_design_matrix_3points(rotated_points, degree)

        # Append the design matrix for this rotation
        augmented_matrices.append(design_matrix)
    
    return augmented_matrices

def weighted_least_squares(augmented_matrices, weights, y):

    # Validate inputs
    if len(augmented_matrices) != len(weights):
        raise ValueError("Number of matrices and weights must be the same.")
    
    # Initialize weighted A and y
    A_weighted = []
    y_weighted = []

    # Construct weighted matrices
    for A, w in zip(augmented_matrices, weights):
        sqrt_w = np.sqrt(w)  # Take the square root of the weight
        A_weighted.append(sqrt_w * A)  # Scale the design matrix
        y_weighted.append(sqrt_w * y)  # Scale the target vector

    # Stack the weighted contributions
    A_weighted = np.vstack(A_weighted)
    y_weighted = np.hstack(y_weighted)

    # Solve the least squares problem: A_weighted.T @ A_weighted @ beta = A_weighted.T @ y_weighted
    beta = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)[0]

    return beta

def evaluate_spherical_polynomial(triple_point, coefficients, degree):

    # Ensure triple_point has the correct shape
    triple_point = np.asarray(triple_point)
    if triple_point.shape != (3, 3):
        raise ValueError("triple_point must have shape (3, 3), representing three points on the sphere.")

    # Convert points to spherical angles
    angles = cartesian_to_spherical(triple_point)  # Shape (3, 2)
    theta, phi = angles[:, 0], angles[:, 1]

    # Generate spherical harmonic terms
    terms = sph_harm_terms(degree)

    # Compute the polynomial value
    value = 0
    for idx, ((l1, m1), (l2, m2), (l3, m3)) in enumerate(terms):
        # Evaluate spherical harmonics for the respective points
        Y1 = sph_harm(m1, l1, phi[0], theta[0])  # For the first point
        Y2 = sph_harm(m2, l2, phi[1], theta[1])  # For the second point
        Y3 = sph_harm(m3, l3, phi[2], theta[2])  # For the third point

        # Triple product
        triple_product = Y1 * Y2 * Y3

        # Accumulate weighted contribution
        value += coefficients[idx] * triple_product

    return value

def evaluate_psym(triple_point, coefficients, degree, rotation_matrices, weights):

    # Ensure weights and rotation_matrices have the same length
    if len(weights) != len(rotation_matrices):
        raise ValueError("The number of weights must match the number of rotation matrices.")

    # Initialize the symmetrized value
    psym_value = 0

    # Loop over rotation matrices and weights
    for R, w in zip(rotation_matrices, weights):
        # Apply the rotation matrix to the triple point
        rotated_point = np.array([R @ p for p in triple_point])

        # Evaluate the polynomial at the rotated point
        f_value = evaluate_spherical_polynomial(rotated_point, coefficients, degree)

        # Accumulate the weighted contribution
        psym_value += w * f_value

    # Return the symmetrized value
    return psym_value

def compute_max_difference_psym(test_points, coefficients, degree, rotation_matrices, weights):

    max_difference = 0

    for triple_point in test_points:
        # Evaluate the polynomial directly
        poly_value = evaluate_spherical_polynomial(triple_point, coefficients, degree)

        # Evaluate the symmetrized version (psym)
        psym_value = evaluate_psym(triple_point, coefficients, degree, rotation_matrices, weights)

        # Compute the difference
        difference = abs(poly_value - psym_value)
        max_difference = max(max_difference, difference)

    return max_difference


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



if __name__ == "__main__":
    max_differences = []
    degree = 3
    for points in range(10, 100, 10):
        filename = f"/Users/hk/RAMathUBC/Quad_SO3/N{degree}.dat"
        rotation_matrices, weights = read_so3_quadrature_file(filename)

        initial_points = np.array([sample_concentrated_sphere(3) for _ in range(points)])
        
        gram_matrices = np.array([gram_matrix_3d(p) for p in initial_points])
        y_initial = f_3d_efficient(gram_matrices)

        augmented_matrices = augmented_design_matrices(initial_points, degree, rotation_matrices)

        coefficients = weighted_least_squares(augmented_matrices, weights, y_initial)

        test_points = np.array([sample_concentrated_sphere(3) for _ in range(30)])

        max_diff = compute_max_difference_psym(test_points, coefficients, degree, rotation_matrices, weights)
        max_differences.append((degree, max_diff))
        
        
    plt.figure()
    plt.yscale("log")
    plt.plot([points for points in range(10,100,10)], np.array(max_differences)[:,1], marker="o")
    plt.title("Max Difference vs Number of fitting points")
    plt.xlabel("Number of Fitting points")
    plt.ylabel("Max difference")
    plt.grid()
    plt.show()

    # Print results
    for degree, max_diff in max_differences:
        print(f"Degree: {degree}, Max Difference: {max_diff}")



