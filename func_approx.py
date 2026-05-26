#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:26:39 2024

@author: hk
"""

import numpy as np
from scipy.linalg import expm
from scipy.special import sph_harm


def sample_uniform_sphere(N):
    """Samples N points uniformly from the unit sphere in 3D."""
    points = np.random.normal(size=(N, 3))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return points


def compute_gram_matrix(points):
    """Computes the N x N Gram matrix for a set of points on the unit sphere."""
    return np.dot(points, points.T)


def trace_exponential_gram_matrix(points):
    """Computes the trace of the exponential of the Gram matrix for a set of points."""
    gram_matrix = np.dot(points, points.T)
    exp_gram_matrix = expm(gram_matrix)
    trace = np.trace(exp_gram_matrix)
    return trace


def spherical_harmonics_design_matrix(points, d):
    """Constructs the design matrix for spherical harmonics up to degree d."""
    theta = np.arccos(points[:, 2])  # polar angle
    phi = np.arctan2(points[:, 1], points[:, 0])  # azimuthal angle
    harmonics = []
    for l in range(d + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            harmonics.append(Y_lm.real)
    return np.column_stack(harmonics)


def least_squares_spherical_harmonics(points, values, d):
    """Finds the least-squares approximation of spherical harmonics up to order d."""
    A = spherical_harmonics_design_matrix(points, d)
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    return coeffs


def evaluate_least_squares_polynomial_at_points(points, coefficients, d):
    """Evaluates the least-squares polynomial at given points."""
    A = spherical_harmonics_design_matrix(points, d)
    evaluated_values = A @ coefficients
    return evaluated_values


def compute_l2_error(coefficients, d, N):
    """Computes the L2 error between the predicted and actual trace of Exp(G) for new points."""
    new_points = sample_uniform_sphere(N)
    trace_exp_gram = trace_exponential_gram_matrix(new_points)
    predicted_values = evaluate_least_squares_polynomial_at_points(new_points, coefficients, d)
    l2_error = np.sqrt(np.mean((predicted_values - trace_exp_gram) ** 2))
    return l2_error


# Number of training and test points
N_particles = 3
N_train = 25
N_test = 25
d = 10  # Degree for spherical harmonics

# Step 1: Sample N_particles training points uniformly on the unit sphere
train_points = np.array([sample_uniform_sphere(N_particles) for _ in range(N_train)])

# Step 2: Generate random function values for these training points
values = np.array([trace_exponential_gram_matrix(i) for i in train_points])


# Step 3: Compute least-squares coefficients
coefficients = least_squares_spherical_harmonics(train_points, values, d)

# Step 4: Compute the L2 error for the predicted values on new points
l2_error = compute_l2_error(coefficients, d, N_test)

# Print the L2 error
print(f"L2 error between predicted values and trace of Exp(G) for {N_test} points (d={d}):", l2_error)
