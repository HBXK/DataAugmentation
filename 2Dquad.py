#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:25:38 2024

@author: hk
"""
import numpy as np
from scipy.linalg import expm, eigh
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from itertools import permutations

plt.rcParams.update({
    "axes.titlesize": 17,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
})


def points_circle(N):
    return np.random.uniform(0, 2 * np.pi, (N, 3))

def points_von_mises(N, kappa):

    return np.mod(np.random.vonmises(0, kappa, size = (N,3)),2*np.pi) 

def points_fix_0(N):
    thetas = np.zeros((N, 3))

    thetas[:, 1:] = np.random.uniform(0,2*np.pi, (N,2))

    return thetas

def points_perturbed_0(N, eps = 100):
    thetas = points_fix_0(N)
    thetas[:,0] = np.mod(np.random.vonmises(0, eps, size = (N,)),2*np.pi)
    

    return thetas



def xgrid(N):
    return np.arange(2*N)*np.pi/N

def kgrid(N):
    return np.concatenate((np.arange(N), np.arange(-N,0)))


def tensor_xgrid(N, d):
    axis = xgrid(N)
    grids = np.meshgrid(*([axis] * d), indexing="ij")   # d arrays
    return np.stack(grids, axis=-1) 

def fft(F, N, d, degree, valid_powers, upsample=2):
    n = (N,)*d
    n = tuple(int(k) for k in n)
    m = tuple(k * upsample for k in n) 
    coords = tensor_xgrid(N, d)
    f_vals = np.asarray(np.apply_along_axis(F, -1, coords, degree, valid_powers))
    coeffs = fftn(f_vals)
    if upsample > 1:
        coeffs = fftshift(coeffs)                       # move k=0 to centre
        pad_width = [((mj - nj)//2, (mj - nj + 1)//2)
                     for nj, mj in zip(n, m)]           # symmetric pad
        coeffs = np.pad(coeffs, pad_width, mode="constant")
        coeffs = ifftshift(coeffs)                      # back to FFT order
    interp_vals = ifftn(coeffs) * (upsample ** d)       # rescale amplitudes
    interp_vals = interp_vals.real                      # f is real‑valued
    axis_fine = [np.linspace(-np.pi, np.pi, k)
                 for k in m]
    coords_fine = np.meshgrid(*axis_fine, indexing="ij")

    return interp_vals, coords_fine


def gram_matrix(N, points):

    gram_matrices = np.zeros((N, 3, 3))

    for i in range(N):
        angles = points[i]
        for j in range(3):
            for k in range(3):
                gram_matrices[i, j, k] = np.cos(angles[j] - angles[k])

    return gram_matrices


def matrix_exponential_diagonalization(G):

    eigenvalues, eigenvectors = eigh(G)  # Compute eigenvalues and eigenvectors
    exp_diag = np.diag(np.exp(eigenvalues))  # Exponential of the eigenvalues
    return exp_diag   # Reconstruct the exponential


def f(N, Gs):
    traces = np.zeros(N)
    for i in range(N):
        exp_G = matrix_exponential_diagonalization(Gs[i])
        traces[i] = np.trace(exp_G)
    return traces



def F(thetas, degree, valid_powers, alpha = 2):
    valid_powers = np.array(valid_powers)
    # ck = np.zeros((len(valid_powers),))
    # i=0
    # for k in valid_powers:
    #     ck[i] = rng[i]*np.exp(-alpha*sum(abs(ki) for ki in k))
    #     i+=1
    # z = np.exp(1j*thetas)
    # return sum(ck[i]*np.cos(np.prod(z**valid_powers[i])) for i in range(len(valid_powers)))
    ck = rng[:len(valid_powers)] * np.exp(-alpha * np.linalg.norm(valid_powers, axis=1))
    phase = valid_powers @ thetas  # shape: (|K|,) because (|K|,3) @ (3,) => (|K|,)
    complex_exps = np.exp(1j * phase)  # shape (|K|,)
    cos_terms = complex_exps.real
    return np.sum(ck * cos_terms)

def F_trunc(thetas, degree, valid_powersF, valid_powers, alpha = 2):
    valid_powers = np.array(valid_powers)
    valid_powersF = np.array(valid_powersF)
    mask = np.isin(valid_powersF, valid_powers).astype(int)
    indecatrix = np.zeros(len(mask))
    for i in range(len(mask)):
        if np.all(mask[i]==np.array([1,1,1])):
            indecatrix[i] = 1
    # ck = np.zeros((len(valid_powers),))
    # i=0
    # for k in valid_powers:
    #     ck[i] = rng[i]*np.exp(-alpha*sum(abs(ki) for ki in k))
    #     i+=1
    # z = np.exp(1j*thetas)
    # return sum(ck[i]*np.cos(np.prod(z**valid_powers[i])) for i in range(len(valid_powers)))
    ck = rng[:len(valid_powersF)] * np.exp(-alpha * np.linalg.norm(valid_powersF, axis=1))*indecatrix
    phase = valid_powersF @ thetas  # shape: (|K|,) because (|K|,3) @ (3,) => (|K|,)
    complex_exps = np.exp(1j * phase)  # shape (|K|,)
    cos_terms = complex_exps.real
    return np.sum(ck * cos_terms)

def design_matrix_unit_circle(thetas, N, valid_powers):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    z = np.exp(1j * thetas)
    valid_powers = np.array(valid_powers)
    # Compute each term in the polynomial using broadcasting
    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)

    if N == 1:
        return design_matrix
    # Augment the data with rotations by 2*pi/k for k = 1, ..., degree
    # augmented_matrix = []
    # for k in range(1, degree + 1):
    #     rotation = 2 * np.pi*k / (degree+1)
    #     for group in thetas:
    #         rotated_group = group + rotation  # Apply the rotation
    #         z = np.exp(1j * rotated_group)  # Compute z = e^(i * (theta + rotation))
    #         row = [np.prod(z**k) for k in valid_powers]  # Compute each term in the polynomial
    #         augmented_matrix.append(row)

    # # Combine original and augmented matrices
    # design_matrix = np.vstack([design_matrix, augmented_matrix])
    
    k_values = np.arange(N)[:, None, None]
    rotations = 2 * np.pi * k_values / (N)
    rotated_thetas = thetas[None, :, :] + rotations
    
    z_rotated = np.exp(1j * rotated_thetas)
    augmented_matrix = np.prod(z_rotated[:, :, None, :] ** valid_powers[None, None, :, :], axis=3)
    
    return np.vstack([augmented_matrix[i] for i in range(N)])
    



def design_matrix_orig(thetas, degree, valid_powers):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    #modified for invariant representation
    valid_powers = np.array(valid_powers)
    # Compute the design matrix
    z = np.exp(1j * thetas)
    # perm_indices = np.array(list(permutations([0, 1, 2])))  # shape (6, 3)

    # thetas_perm = thetas[:, None, :]  # shape (N, 1, 3)
    # thetas_permuted = np.take_along_axis(thetas_perm, perm_indices[None, :, :], axis=2)  # shape (N, 6, 3)
    # z = np.sum(np.exp(1j * thetas_permuted), axis=1)
    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)
    return np.array(design_matrix)

# def design_matrix_orig(thetas, degree, valid_powers):
#     thetas = np.asarray(thetas)

#     if thetas.shape[1] != 3:
#         raise ValueError("The input angles must have shape (N, 3)")

#     valid_powers = np.array(valid_powers)  # shape (M, 3)
#     N, M = thetas.shape[0], valid_powers.shape[0]

#     # All 6 permutations of [0, 1, 2]
#     perm_indices = np.array(list(permutations([0, 1, 2])))  # shape (6, 3)

#     # Step 1: Permute thetas → shape (N, 6, 3)
#     thetas_exp = thetas[:, None, :]
#     thetas_permuted = np.take_along_axis(thetas_exp, perm_indices[None, :, :], axis=2)

#     # Step 2: Expand to (N, 6, M, 3) to broadcast with valid powers
#     thetas_perm_expanded = thetas_permuted[:, :, None, :]  # (N, 6, 1, 3)
#     powers_expanded = valid_powers[None, None, :, :]       # (1, 1, M, 3)

#     # Step 3: Safely compute powered values
#     # Avoid invalid values for 0 ** negative by masking
#     with np.errstate(divide='ignore', invalid='ignore'):
#         # Compute elementwise: theta^k only where valid
#         powered = np.where(
#             (thetas_perm_expanded == 0) & (powers_expanded < 0),
#             0.0,  # Placeholder value; will get masked in final product
#             thetas_perm_expanded ** powers_expanded
#         )

#     # Step 4: Exponentiation
#     exp_term = np.exp(1j * powered)  # shape (N, 6, M, 3)

#     # Step 5: Product across 3 variables, then sum over permutations
#     term_product = np.prod(exp_term, axis=3)  # (N, 6, M)
#     design_matrix = np.sum(term_product, axis=1)  # (N, M)

#     return design_matrix





def design_matrix_full(thetas, degree, valid_powers):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")
    
    valid_powers = np.array(valid_powers)
    
    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    # valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
    # perm = np.array(list(permutations(thetas)))

    # Compute z = e^(i * theta) for all groups at once
    z = np.exp(1j * thetas)
    perm_indices = np.array(list(permutations([0, 1, 2])))  # shape (6, 3)

    
    # Compute each term in the polynomial using broadcasting
    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)

    return np.array(design_matrix)

def design_matrix_proj(thetas, degree, valid_powers):
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3)")

    valid_powers = np.array(valid_powers)  # shape (M, 3)
    N = thetas.shape[0]
    
    # Compute z = e^(i * theta) for all groups at once
    z = np.exp(1j * thetas)  # shape (N, 3)

    # Compute each term: z^k, shape (N, M)
    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)  # shape (N, M)

    # Zero out entries where sum(k) != 0
    power_sums = np.sum(valid_powers, axis=1)  # shape (M,)
    mask = power_sums == 0                    # shape (M,)
    design_matrix[:, ~mask] = 0

    return design_matrix



# def design_matrix_full(thetas, degree, valid_powers):
#     thetas = np.asarray(thetas)

#     if thetas.shape[1] != 3:
#         raise ValueError("The input angles must have shape (N, 3)")

#     valid_powers = np.array(valid_powers)  # shape (M, 3)
#     N, M = thetas.shape[0], valid_powers.shape[0]

#     # All 6 permutations of [0, 1, 2]
#     perm_indices = np.array(list(permutations([0, 1, 2])))  # shape (6, 3)

#     # Step 1: Permute thetas → shape (N, 6, 3)
#     thetas_exp = thetas[:, None, :]
#     thetas_permuted = np.take_along_axis(thetas_exp, perm_indices[None, :, :], axis=2)

#     # Step 2: Expand to (N, 6, M, 3) to broadcast with valid powers
#     thetas_perm_expanded = thetas_permuted[:, :, None, :]  # (N, 6, 1, 3)
#     powers_expanded = valid_powers[None, None, :, :]       # (1, 1, M, 3)

#     # Step 3: Safely compute powered values
#     # Avoid invalid values for 0 ** negative by masking
#     with np.errstate(divide='ignore', invalid='ignore'):
#         # Compute elementwise: theta^k only where valid
#         powered = np.where(
#             (thetas_perm_expanded == 0) & (powers_expanded < 0),
#             0.0,  # Placeholder value; will get masked in final product
#             thetas_perm_expanded ** powers_expanded
#         )

#     # Step 4: Exponentiation
#     exp_term = np.exp(1j * powered)  # shape (N, 6, M, 3)

#     # Step 5: Product across 3 variables, then sum over permutations
#     term_product = np.prod(exp_term, axis=3)  # (N, 6, M)
#     design_matrix = np.sum(term_product, axis=1)  # (N, M)

#     return design_matrix





def design_matrix_MC(thetas, degree, nb, valid_powers):

    # Ensure thetas is a numpy array
    thetas = np.asarray(thetas)

    if thetas.shape[1] != 3:
        raise ValueError("The input angles must have shape (N, 3) for groups of three angles.")

    # Generate all possible combinations of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    valid_powers = np.array(valid_powers)
    z = np.exp(1j * thetas)

    # Compute the design matrix
    design_matrix = np.prod(z[:, None, :] ** valid_powers[None, :, :], axis=2)
    if nb != 1:
        rotations = np.random.uniform(0,2*np.pi, (nb-1, ))[:, None, None]
        rotated_thetas = thetas[None, :, :] + rotations
   
        z_rotated = np.exp(1j * rotated_thetas)
        augmented_matrix = np.prod(z_rotated[:, :, None, :] ** valid_powers[None, None, :, :], axis=3)
   
        return np.vstack([design_matrix] + [augmented_matrix[i] for i in range(nb-1)])
   
    else:
        return design_matrix

def augment_target_values(y, N):
    if N == 1:
        return y
    y = np.asarray(y)
    # Replicate y (degree + 1) times: 1 original + degree augmentations
    augmented_y = np.tile(y, N)
    return augmented_y


# def least_square(X, y):

#     X_conj_transpose = np.conjugate(X.T)  
#     w = np.linalg.solve(X_conj_transpose @ X, X_conj_transpose @ y)
#     return w

def least_square(X,y, tol = 0):
    if tol == 0:
        return np.linalg.lstsq(X, y)[0]
    U,R,V = np.linalg.svd(X)
    r = R[R>tol]
    rinv = np.diag(np.linalg.pinv(np.diag(r)))
    sigma = np.zeros((V.shape[0], U.shape[0]))
    np.fill_diagonal(sigma, rinv)
    return (V.T @ sigma @ U.T)@y
    
    
    

def least_squares_qr(X, y):

    # Perform full QR decomposition of X
    Q, R_full = np.linalg.qr(X, mode='complete')

    # Extract the upper triangular part R (n x n)
    n = X.shape[0]
    R = R_full[::, :n]  # Take only the first n rows of R_full

    # Compute Q^T * y
    QTy = np.dot(Q.T, y)
    
    # Solve R * beta = QTy_relevant
    beta = np.linalg.solve(R, QTy)
    
    return beta




def evaluate_polynomial(theta1, theta2, theta3, coefficients, degree, valid_powers):

    # Compute z = e^(i * theta) for the input angles
    z = np.exp(1j * np.array([theta1, theta2, theta3]))

    # Generate all valid powers of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    # valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]

    # Evaluate the polynomial
    valid_powers = np.array(valid_powers)
    
    value = np.sum(coefficients * np.prod(z**valid_powers, axis=1))

    return value

def eval_poly_invar(theta1, theta2, theta3, coefficients, degree, valid_powers):
    # Compute z = e^(i * theta) for the input angles
    z = np.exp(1j * np.array([theta1, theta2, theta3]))

    # Generate all valid powers of (k1, k2, k3) with sum(|k1| + |k2| + |k3|) == degree
    valid_powers = np.array(valid_powers)

    # Evaluate the polynomial
    value = np.sum(coefficients * np.prod(z**valid_powers, axis=1))

    return value




def psym(theta1,theta2,theta3, coefficients, degree, valid_powers):
    sym_value = 0
    for k in range(degree):
        rotation = 2 * np.pi*k / (degree)

        rotated_theta1 = theta1 + rotation
        rotated_theta2 = theta2 + rotation
        rotated_theta3 = theta3 + rotation

        rotated_value = evaluate_polynomial(rotated_theta1, rotated_theta2, rotated_theta3, coefficients, degree, valid_powers)

        sym_value += rotated_value

    sym_value /= degree

    return sym_value

def psym_MC(theta1, theta2, theta3, coefficients, degree, nb):

    sym_value = 0
    for k in range(nb):
        rotation = rotations[k]

        rotated_theta1 = theta1 + rotation
        rotated_theta2 = theta2 + rotation
        rotated_theta3 = theta3 + rotation

        rotated_value = evaluate_polynomial(rotated_theta1, rotated_theta2, rotated_theta3, coefficients, degree)

        sym_value += rotated_value

    sym_value /= degree

    return sym_value


def compute_max_difference(N, degree, nb):

    # Generate points and compute the Gram matrix
    thetas = points_circle(N)
    gram_matrices = gram_matrix(N, thetas)

    # Compute target values
    y = f(N,gram_matrices)

    # Create the design matrix and augment the target values
    X = design_matrix_unit_circle(thetas, degree)
    #X = design_matrix_MC(thetas, degree, nb)

    augmented_y = augment_target_values(y, nb)

    # Compute least squares coefficients
    coefficients = least_square(X, augmented_y)

    # Randomly sample 30 more points
    test_points = points_circle(30)

    # Compute the max difference between psym and evaluate_polynomial
    max_difference = 0
    for theta1, theta2, theta3 in test_points:
        psym_value = psym(theta1, theta2, theta3, coefficients, degree)
        poly_value = evaluate_polynomial(theta1, theta2, theta3, coefficients, degree)
        difference = abs(psym_value - poly_value)
        max_difference = max(max_difference, difference)

    return max_difference





rng = np.random.uniform(-1,1, size=(50000,))


# Poly Truncation vs High data LSQ

if __name__!= "__main__":

    degrees = [i for i in range(3,16, 3)]
    degreeF = 30
    test_points = points_circle(15)
    best_invar_lsq = []
    best_full_lsq = []
    truncation_error  = []
    proj_error = []
    N=5000
    # tol = -1
    thetas = points_perturbed_0(N)
    valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
    y_target = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
    plt.figure()
    plt.yscale("log")
    plt.ylim(1e-9,1e1)
    # plt.title("Fitting Error of Non-Augmented LSQ (D3)")
    plt.xlabel("Degree")
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for degree in tqdm(degrees):
        max_differences = []
        valid_powers_invar = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree and sum(ki for ki in k)==0]
        valid_powers_full = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]

        DM_test = design_matrix_proj(test_points, degree, valid_powers_full)

        X_invar = design_matrix_orig(thetas, degree, valid_powers_invar)
        X_full = design_matrix_full(thetas, degree, valid_powers_full)
        coeff_full = np.linalg.lstsq(X_full, y_target, rcond=10**-4.5)[0]
        coeff_invar = np.linalg.lstsq(X_invar, y_target, rcond=10**-4.5)[0]
        poly_values_invar =[] 
        poly_values_full = []
        for theta1, theta2, theta3 in test_points:
            poly_values_invar.append(eval_poly_invar(theta1, theta2, theta3, coeff_invar, degree, valid_powers_invar))
            poly_values_full.append(evaluate_polynomial(theta1, theta2, theta3, coeff_full, degree, valid_powers_full))
        y_truncation = np.apply_along_axis(F_trunc, 1, test_points, degree, valid_powersF, valid_powers_invar)
        y_test = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)
        truncation_error.append(1/np.sqrt(15)*np.linalg.norm(y_truncation-y_test))
        best_invar_lsq.append(1/np.sqrt(15)*np.linalg.norm(poly_values_invar-y_test))
        best_full_lsq.append(1/np.sqrt(15)*np.linalg.norm(poly_values_full-y_test))
        proj_error.append(1/np.sqrt(15)*np.linalg.norm(DM_test @ coeff_full-y_test))

        # plt.plot([i for i in range(10,1520,100)], max_differences, marker="o", label=f'{degree}')
    plt.plot(degrees, best_invar_lsq, marker="o", label='LSQ for invariant basis')
    plt.plot(degrees, truncation_error, marker="o", label='Truncated Polynomial')
    plt.plot(degrees, best_full_lsq, marker="o", label='LSQ for full basis')
    plt.plot(degrees, proj_error, marker="o", label='LSQ for full basis+ proj')

    # plt.plot(degrees, np.exp(-0.40832958*np.array(degrees)), label=f'Exp fit')
    plt.legend()
    plt.grid()
    plt.show()




# Find sigma_min


if __name__!= "__main__":
    plt.figure()
    plt.title("Best fit vs sigma_min")
    plt.xlabel("sigma_min")
    plt.ylabel("L2 Error")
    plt.yscale("log")
    plt.xscale("log")
    degree = 4
    degreeF = 30
    test_points = points_circle(20)
    best_invar_lsq = []
    best_full_lsq = []
    truncation_error  = []
    proj_error = []
    N=5000
    tols = 10**(np.array([-2,-2.5,-3,-3.5,-4, -4.5,-5,-5.5,-6,-6.5,-7,-7.5,-8,-8.5]))
    thetas = points_perturbed_0(N)
    valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
    y_target = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
    valid_powers_full = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
    y_test = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)
    X_full = design_matrix_full(thetas, degree, valid_powers_full)
    for tol in tqdm(tols):
        max_differences = []


        coeff_full = np.linalg.lstsq(X_full, y_target, tol)[0]
        poly_values_full = []
        for theta1, theta2, theta3 in test_points:
            poly_values_full.append(evaluate_polynomial(theta1, theta2, theta3, coeff_full, degree, valid_powers_full))
        best_full_lsq.append(1/np.sqrt(20)*np.linalg.norm(poly_values_full-y_test))

        # plt.plot([i for i in range(10,1520,100)], max_differences, marker="o", label=f'{degree}')
    plt.plot(tols, best_full_lsq, marker="o", label='LSQ for varying sigma_min')

    # plt.plot(degrees, np.exp(-0.40832958*np.array(degrees)), label=f'Exp fit')
    plt.legend()
    plt.grid()
    plt.show()


# Quality of Approximation with rotated data









# Augment v Augment + proj

if __name__!= "__main__":
    plt.figure()
    plt.yscale("log")
    plt.title("Augmenting data with projection(fixed 0)")
    plt.xlabel("Degree")
    plt.ylabel("L2 Error")
    degree = 7
    degreeF = 30
    test_points = points_circle(15)
    quad_simple = []
    quad_proj = []
    MC_simple  = []
    MC_proj = []
    invar = []
    # tol = -1
    thetas = points_fix_0(50)
    valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]

    valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
    DM_test = design_matrix_proj(test_points, degree, valid_powers)
    valid_powers_invar = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree and sum(ki for ki in k)==0]

    for N in tqdm([1,3,5,7,9,12]):
        max_differences = []


        X_invar = design_matrix_orig(thetas, degree, valid_powers_invar)
        X_quad = design_matrix_unit_circle(thetas, N, valid_powers)
        X_MC = design_matrix_MC(thetas, degree, N, valid_powers)       
        y_target = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
        coeff_invar = least_square(X_invar, y_target)

        y_target = augment_target_values(y_target, N)

        coeff_quad = least_square(X_quad, y_target)
        coeff_MC = least_square(X_MC, y_target)
        poly_values_invar =[] 
        poly_values_quad = []
        poly_values_MC = []
        for theta1, theta2, theta3 in test_points:
            poly_values_invar.append(eval_poly_invar(theta1, theta2, theta3, coeff_invar, degree, valid_powers_invar))
            poly_values_quad.append(evaluate_polynomial(theta1, theta2, theta3, coeff_quad, degree, valid_powers))
            poly_values_MC.append(evaluate_polynomial(theta1, theta2, theta3, coeff_MC, degree, valid_powers))
        y_test = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)
        invar.append(1/np.sqrt(15)*np.linalg.norm(poly_values_invar-y_test))
        quad_simple.append(1/np.sqrt(15)*np.linalg.norm(poly_values_quad-y_test))
        quad_proj.append(1/np.sqrt(15)*np.linalg.norm(DM_test @ coeff_quad-y_test))
        MC_simple.append(1/np.sqrt(15)*np.linalg.norm(poly_values_MC-y_test))
        MC_proj.append(1/np.sqrt(15)*np.linalg.norm(DM_test @ coeff_MC-y_test))

    plt.plot([1,3,5,7,9,12], quad_simple, marker="o", label='Quad')
    plt.plot([1,3,5,7,9,12], quad_proj, marker="o", label='Quad+proj')
    plt.plot([1,3,5,7,9,12], MC_simple, marker="o", label='MC')
    plt.plot([1,3,5,7,9,12], MC_proj, marker="o", label='MC+proj')
    plt.plot([1,3,5,7,9,12], invar, marker="o", label='Invariant')

    # plt.plot(degrees, np.exp(-0.40832958*np.array(degrees)), label=f'Exp fit')
    plt.legend()
    plt.grid()
    plt.show()


if __name__!= "__main__":
    plt.figure()
    plt.yscale("log")
    plt.xscale("log")

    plt.title("Fitting error with quadrature augmentation, L=3, D3")
    plt.xlabel("Nb of Rotation")
    plt.ylabel("L2 Error")
    degree = 3
    degreeF = 30
    test_points = points_circle(15)
    quad_simple = []
    quad_proj = []
    MC_simple  = []
    MC_proj = []
    # tol = -1
    thetas = points_perturbed_0(100)
    valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
    valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
    y_test = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)

    y_target = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)

    DM_test = design_matrix_proj(test_points, degree, valid_powers)
    valid_powers_invar = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree and sum(ki for ki in k)==0]
    rotn = np.array([1,2,4,6,8,10,12,15,20,40,70,100])
    X_invar = design_matrix_orig(thetas, degree, valid_powers_invar)
    coeff_invar = least_square(X_invar, y_target)[0]
    poly_values_invar =[] 

    for theta1, theta2, theta3 in test_points:
        poly_values_invar.append(eval_poly_invar(theta1, theta2, theta3, coeff_invar, degree, valid_powers_invar))
    invar = (1/np.sqrt(15)*np.linalg.norm(poly_values_invar-y_test))

    for N in tqdm(rotn):
        max_differences = []


        X_quad = design_matrix_unit_circle(thetas, N, valid_powers)
        X_MC = design_matrix_MC(thetas, degree, N, valid_powers)       
        y_target = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)

        y_target = augment_target_values(y_target, N)

        coeff_quad = least_square(X_quad, y_target, tol=10**-4.5)
        coeff_MC = least_square(X_MC, y_target, tol=10**-4.5)
        poly_values_quad = []
        poly_values_MC = []
        for theta1, theta2, theta3 in test_points:
            poly_values_quad.append(evaluate_polynomial(theta1, theta2, theta3, coeff_quad, degree, valid_powers))
            poly_values_MC.append(evaluate_polynomial(theta1, theta2, theta3, coeff_MC, degree, valid_powers))
        quad_simple.append(1/np.sqrt(15)*np.linalg.norm(poly_values_quad-y_test))
        quad_proj.append(1/np.sqrt(15)*np.linalg.norm(DM_test @ coeff_quad-y_test))
        MC_simple.append(1/np.sqrt(15)*np.linalg.norm(poly_values_MC-y_test))
        MC_proj.append(1/np.sqrt(15)*np.linalg.norm(DM_test @ coeff_MC-y_test))


    plt.plot (rotn, [invar for i in rotn], linestyle = '--', label = f'Invar')
    plt.plot(rotn, quad_simple, marker="o", label='Quad')
    plt.plot(rotn, quad_proj, marker="o", label='Quad+proj')
    plt.plot(rotn, MC_simple, marker="o", label='MC')
    plt.plot(rotn, MC_proj, marker="o", label='MC+proj')    
    # plt.plot(rotn, invar, marker="o", label='Invariant')

    # plt.plot(degrees, np.exp(-0.40832958*np.array(degrees)), label=f'Exp fit')
    plt.legend()
    plt.grid()
    plt.show()


























# Different Psym
if __name__!= "__main__":

    degrees = [3,6,9]
    degreeF = 30
    test_points = points_circle(15)
    valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
    thetas = points_perturbed_0(100)
    plt.figure()
    plt.yscale("log")
    # plt.title("Quadrature rules and εsym, D3, k=100, tol=1e-4, np")
    plt.xlabel("Number of Rotations")
    # plt.ylabel("||f - sym(f)||")
    plt.ylim(1e-17,1e0)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    for degree in degrees:
        quad_psym = []
        MC_psym = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        for N in tqdm([k for k in range(1,16,1)]):
            psym_quad = []
            poly_quad = []
            # psym_MC_error = []
            # poly_MC = []
            X_quad = design_matrix_unit_circle(thetas, N, valid_powers)
            # X_MC = design_matrix_MC(thetas, degree, N, valid_powers)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            y = augment_target_values(y, N)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff_quad = np.linalg.lstsq(X_quad, y, rcond=1e-4)[0]
            # coeff_MC = least_square(X_MC, y)[0]
            max_difference = 0
            for theta1, theta2, theta3 in test_points:
                psym_quad.append(psym(theta1, theta2, theta3, coeff_quad, degree, valid_powers))
                poly_quad.append(evaluate_polynomial(theta1, theta2, theta3, coeff_quad, degree, valid_powers))
                # psym_MC_error.append(psym(theta1, theta2, theta3, coeff_MC, degree, valid_powers))
                # poly_MC.append(evaluate_polynomial(theta1, theta2, theta3, coeff_MC, degree, valid_powers))
            psym_quad = np.array(psym_quad)
            poly_quad = np.array(poly_quad)
            # psym_MC_error = np.array(psym_MC_error)
            # poly_MC = np.array(poly_MC)
            quad_psym.append(1/np.sqrt(15)*np.linalg.norm(psym_quad-poly_quad))
            # MC_psym.append(1/(15)*np.linalg.norm(psym_MC_error-poly_MC))
        plt.plot([k for k in range(1,16,1)], quad_psym, marker = "o", label = f'Degree : {degree}')
        # plt.plot([k for k in range(1,13,1)], MC_psym, marker = "o", label = f'MC : {degree}')

    # plt.plot(degrees, np.exp(-0.40832958*np.array(degrees)), label=f'Exp fit')
    plt.legend()
    plt.grid()
    plt.show()


# Evolution of L2 Error for MC and Quad
if __name__!= "__main__":
    degrees = [3,6,9]
    degreeF = 30
    test_points = points_circle(15)
    valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
    target = np.apply_along_axis(F, 1, test_points, degreeF, valid_powersF)
    


    # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    for degree in degrees:
        quad_l2 = []
        MC_l2 = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        DM_proj = design_matrix_proj(test_points, degree, valid_powers)

        for N in tqdm([k for k in range(1,21,1)]):
            tol = 10**-4.5
            poly_quad = []
            poly_MC = []
            X_quad = design_matrix_unit_circle(thetas, N, valid_powers)
            X_MC = design_matrix_MC(thetas, degree, N, valid_powers)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
            y = augment_target_values(y, N)
            # for theta in thetas:
            #     y.append(F(theta, degreeF))
            # y = np.array(y)
            coeff_quad = np.linalg.lstsq(X_quad, y, tol)[0]
            coeff_MC = np.linalg.lstsq(X_MC, y, tol)[0]
            poly_quad = DM_proj @ coeff_quad
            poly_MC = DM_proj @ coeff_MC
            # max_difference = 0
            # for theta1, theta2, theta3 in test_points:
            #     poly_quad.append(evaluate_polynomial(theta1, theta2, theta3, coeff_quad, degree, valid_powers))
            #     poly_MC.append(evaluate_polynomial(theta1, theta2, theta3, coeff_MC, degree, valid_powers))
            # poly_quad = np.array(poly_quad)
            # poly_MC = np.array(poly_MC)
            quad_l2.append(1/(15)*np.linalg.norm(target-poly_quad))
            MC_l2.append(1/(15)*np.linalg.norm(target-poly_MC))
        plt.plot([k for k in range(1,21,1)], quad_l2, marker = "o", label = f'Quad+P : {degree}')
        plt.plot([k for k in range(1,21,1)], MC_l2, marker = "o", label = f'MC+P : {degree}')

    # plt.plot(degrees, np.exp(-0.40832958*np.array(degrees)), label=f'Exp fit')
    plt.legend(loc='upper right', fontsize='10')
    plt.grid()
    plt.show()


# More data half slope
if __name__ != "__main__":
    plt.figure()
    plt.yscale("log")
    plt.xscale("log")
    plt.title("MC (2D, )")
    plt.xlabel("Data points")
    plt.ylabel("||f-sym f||")
    degrees = [3,6,9]
    degreeF = 15
    test_points = points_circle(15, 3)
    standard_approx = []
    best = []
    valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]
    for degree in degrees:
        max_differences = []
        valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
        for N in tqdm([k for k in range(1000, 11000,1000)]):
            
            thetas = points_von_mises(N, 3)
            X = design_matrix_full(thetas, degree, valid_powers)
            y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)

            coeff = least_square(X, y)
            poly_values =[] 
            psym_MC_error = []

            for theta1, theta2, theta3 in test_points:
                poly_values.append(evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers))
                psym_MC_error.append(psym(theta1, theta2, theta3, coeff, degree, valid_powers))

            max_differences.append(1/np.sqrt(15)*np.linalg.norm(np.array(poly_values)-np.array(psym_MC_error)))
        plt.plot([k for k in range(1000, 11000,1000)], max_differences, marker = "o", label = f'f-sym f, degree : {degree}')
    plt.legend()
    plt.grid()
    plt.show()


# MC data augmentation

if __name__ == "__main__":
    degrees = [3,6,9]
    degreeF = 30
    rots = [4,8,10,12,14,16,32, 64, 128,256]
    valid_powersF = [k for k in product(range(-degreeF, degreeF + 1), repeat=3) if sum(abs(ki) for ki in k) <= degreeF  and sum(ki for ki in k)==0]

    all_psyms = np.zeros((10,len(degrees),len(rots)))
    for dist in [1,2]:
        if dist == 0:
            tol = None
            plt.figure()
            plt.yscale("log")
            plt.xscale("log")

            # plt.title("Symmetrisation Methods")
            plt.xlabel("Number of Rotations")
            plt.ylabel("L2 error")
            plt.ylim(1e-6,1e0)
        elif dist == 1:
            plt.figure()
            plt.yscale("log")
            plt.xscale("log")

            tol= 10**(-4.5)
            # plt.title("Symmetrisation Methods")
            plt.xlabel("Number of Rotations")
            # plt.ylabel("L2 error")
            plt.ylim(1e-6,1e0)
            plt.xlim(3e0,3e2)
            plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        elif dist == 2:
            tol= 10**(-4.5)
            plt.figure()
            plt.yscale("log")
            plt.xscale("log")

            # plt.title("Symmetrisation Methods")
            plt.xlabel("Number of Rotations")
            # plt.ylabel("L2 error")
            plt.ylim(1e-6,1e0)
            plt.xlim(3e0,3e2)
            plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            
        for i in range(10):
            test_points = points_circle(15)
            if dist == 0:
                thetas = points_circle(100)
            elif dist == 1:
                thetas = points_fix_0(100)
               
            elif dist == 2:
                thetas = points_perturbed_0(100)
                
            for a, degree in enumerate(degrees):
                max_differences = []
                valid_powers = [k for k in product(range(-degree, degree + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree]
                for N in tqdm(rots):
                
                    X = design_matrix_MC(thetas, degree, N, valid_powers)
                    y = np.apply_along_axis(F, 1, thetas, degreeF, valid_powersF)
                    y = augment_target_values(y, N)
                    coeff = np.linalg.lstsq(X, y)[0]
                    poly_values =[] 
                    psym_MC_error = []
    
                    for theta1, theta2, theta3 in test_points:
                        poly_values.append(evaluate_polynomial(theta1, theta2, theta3, coeff, degree, valid_powers))
                        psym_MC_error.append(psym(theta1, theta2, theta3, coeff, degree, valid_powers))
    
                    max_differences.append(1/np.sqrt(15)*np.linalg.norm(np.array(poly_values)-np.array(psym_MC_error)))
                all_psyms[i,a] = max_differences
                # plt.plot(rots, max_differences, marker = "o",  color = "gray", alpha = 0.3)

        np.save(f'C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//MC2d//all_psymsD{dist}.npy', all_psyms)


        for a, degree in enumerate(degrees):
    
            mean_psym = np.mean(all_psyms[:, a, :], axis = 0)
            stvd_psym = np.std(all_psyms[:, a, :], axis =0)
            # plt.plot(rots, mean_psym, marker="", label=f'Mean : {degree}')
            plt.errorbar(rots, mean_psym, yerr = [np.zeros_like(stvd_psym), stvd_psym], linestyle=None, lolims=True, ecolor='black', label=f'Mean : {degree}')
            # plt.fill_between(rots, mean_psym-stvd_psym, mean_psym+stvd_psym, alpha = 0.2,label=f' +/- stdv')
        plt.plot(np.logspace(0,2.5,200), 0.002*np.logspace(0 ,2.5,200)**(-0.5), linestyle = "--",  label = f'T^-1/2')
    
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()



# Conservation of angular momentum

def F_perturbed(thetas, degree, valid_powers, epsilon, alpha = 2):
    valid_powers = np.array(valid_powers)
    # ck = np.zeros((len(valid_powers),))
    # i=0
    # for k in valid_powers:
    #     ck[i] = rng[i]*np.exp(-alpha*sum(abs(ki) for ki in k))
    #     i+=1
    # z = np.exp(1j*thetas)
    # return sum(ck[i]*np.cos(np.prod(z**valid_powers[i])) for i in range(len(valid_powers)))
    ck = rng[:len(valid_powers)] * np.exp(-alpha * np.linalg.norm(valid_powers, axis=1))
    phase = valid_powers @ thetas  # shape: (|K|,) because (|K|,3) @ (3,) => (|K|,)
    complex_exps = np.exp(1j * phase)  # shape (|K|,)
    cos_terms = complex_exps.real
    return np.sum(ck * cos_terms)+ epsilon*np.cos(thetas[0])*np.cos(thetas[1])*np.cos(thetas[1])

def psym_perturbed(thetas, degree, valid_powers, epsilon, alpha = 2):
    sym_value = 0
    for i in range(degree+1):
        
        R_thetas = np.array([thetas[0]+2*i*np.pi/(degree+1), thetas[1]+2*i*np.pi/(degree+1), thetas[2]+2*i*np.pi/(degree+1)])
        sym_value +=  F_perturbed(R_thetas, degree, valid_powers, epsilon)

    sym_value /= (degree+1)
    return sym_value




def J(omega: np.ndarray) -> float:
    """Return J = Σ_i ω_i."""
    return float(np.sum(omega))

# def grad_f(theta: np.ndarray, Func, degree_potential,valid_powers_potential, epsilon, h: float = 1.0e-6) -> np.ndarray:
#     """
#     Numerical gradient ∇f using leapfrog
#     Args
#     ----
#     theta : (3,) ndarray  – current angles
#     F     : callable      – user-supplied f(θ)  (scalar)
#     h     : float         – stepsize for the finite difference

#     Returns
#     -------
#     ∇f (3,) ndarray
#     """
#     grad = np.zeros_like(theta, dtype=float)
#     for j in range(theta.size):
#         forward = theta.copy(); forward[j] += h
#         backward = theta.copy(); backward[j] -= h
#         grad[j] = (Func(forward, degree_potential,valid_powers_potential, epsilon) - 2* Func(theta, degree_potential,valid_powers_potential, epsilon) + Func(backward, degree_potential,valid_powers_potential, epsilon)) / (2.0 * h)
#     return grad

def grad_f(thetas, degree, valid_powers, epsilon, alpha = 2):
    valid_powers = np.array(valid_powers)
    ck = rng[:len(valid_powers)] * np.exp(-alpha * np.linalg.norm(valid_powers, axis=1))
    phase = valid_powers @ thetas  # shape: (|K|,) because (|K|,3) @ (3,) => (|K|,)
    complex_exps = np.exp(1j * phase)  # shape (|K|,)
    return np.array([np.sum(ck * complex_exps * valid_powers[:, 0]*1j).real - epsilon*np.sin(thetas[0])*np.cos(thetas[1])*np.cos(thetas[2]),
            np.sum(ck * complex_exps * valid_powers[:, 1]*1j).real - epsilon*np.cos(thetas[0])*np.sin(thetas[1])*np.cos(thetas[2]),
            np.sum(ck * complex_exps * valid_powers[:, 2]*1j).real - epsilon*np.cos(thetas[0])*np.cos(thetas[1])*np.sin(thetas[2])])



# ---------- velocity-Verlet integrator --------------------------------------

def verlet_integrate(theta0: np.ndarray,
                     omega0: np.ndarray,
                     dt: float,
                     n_steps: int,
                     valid_powers_potential, degree_potential,
                     epsilon):
    """
    Integrate θ̈ = −∇f(θ) with a velocity-Verlet scheme.

    Parameters
    ----------
    theta0, omega0 : (3,) arrays   – initial angle and angular velocity
    F              : callable      – returns scalar f(θ)
    dt             : float         – time step
    n_steps        : int           – number of iterations

    Returns
    -------
    K_series : (n_steps+1,) ndarray – kinetic energy Σ ω² for each saved step
    """
    theta = theta0.astype(float).copy()
    omega = omega0.astype(float).copy()

    J_series = np.empty(n_steps + 1)
    J_series[0] = J(omega)

    for n in tqdm(range(1, n_steps + 1)):
        # a(t)  ---------------------------------------------------------------
        a      = -grad_f(theta, degree_potential,valid_powers_potential, epsilon)

        # θ(t+dt) -------------------------------------------------------------
        theta += omega * dt + 0.5 * a * dt**2

        # a(t+dt) -------------------------------------------------------------
        a_new  = -grad_f(theta, degree_potential,valid_powers_potential, epsilon)

        # ω(t+dt) -------------------------------------------------------------
        omega += 0.5 * (a + a_new) * dt

        # store K -------------------------------------------------------------
        J_series[n] = J(omega)

    return J_series



def hitting_time(theta0: np.ndarray,
                     omega0: np.ndarray,
                     dt: float,
                     tol:float,
                     valid_powers_potential, degree_potential,
                     epsilon):
    """
    Integrate θ̈ = −∇f(θ) with a velocity-Verlet scheme.

    Parameters
    ----------
    theta0, omega0 : (3,) arrays   – initial angle and angular velocity
    F              : callable      – returns scalar f(θ)
    dt             : float         – time step
    n_steps        : int           – number of iterations

    Returns
    -------
    K_series : (n_steps+1,) ndarray – kinetic energy Σ ω² for each saved step
    """
    theta = theta0.astype(float).copy()
    omega = omega0.astype(float).copy()

    J_series = np.ones(10000000)
    J_series[0] = J(omega)
    n=0
    while np.abs(J_series[n])<=tol and 1.0 in J_series:
        n += 1
        if n % 100000 == 0:
            print(n)
        
        # a(t)  ---------------------------------------------------------------
        a      = -grad_f(theta, degree_potential,valid_powers_potential, epsilon)

        # θ(t+dt) -------------------------------------------------------------
        theta += omega * dt + 0.5 * a * dt**2

        # a(t+dt) -------------------------------------------------------------
        a_new  = -grad_f(theta, degree_potential,valid_powers_potential, epsilon)

        # ω(t+dt) -------------------------------------------------------------
        omega += 0.5 * (a + a_new) * dt

        # store K -------------------------------------------------------------
        J_series[n] = J(omega)
    return n


if __name__ != "__main__":
    # plt.title("Approximate conservation of Angular momentum ")
    # plt.xlabel("||f-sym f||")
    # plt.ylabel("Deviation in Angular Momentum")
    degree_potential = 10
    valid_powers_potential = [k for k in product(range(-degree_potential, degree_potential + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree_potential and sum(ki for ki in k)==0]
    theta0 = np.array([2.10, 0.20, -1.15])
    omega0 = np.array([-0.5, 0.25, 0.25])
    dt      = 5e-2          
    test_points = points_circle(100)
    # J_t_unperturbed = verlet_integrate(theta0, omega0, F_perturbed, dt, n_steps, valid_powers_potential, degree_potential, 0)
    psyms = []
    hitting_times2 = []
    F_unperturbed = np.apply_along_axis(F, 1, test_points, degree_potential, valid_powers_potential)


    for eps in [5e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:


        print('NEW')
        F_p = np.apply_along_axis(F_perturbed, 1, test_points, degree_potential, valid_powers_potential, eps)
        sym = []
        for thetas in test_points:
            sym.append(psym_perturbed(thetas, degree_potential, valid_powers_potential, alpha = 2, epsilon= eps))
        # psym = np.sqrt(1/(2*np.pi*100))*np.linalg.norm(F_p-np.array(sym))
        # psyms.append(np.linalg.norm(F_p-np.array(sym)))
        hitting_times2.append(hitting_time(theta0, omega0, dt, 0.01, valid_powers_potential, degree_potential, eps))


    





if __name__ != "__main__":
    # plt.title("Approximate conservation of Angular momentum ")
    # plt.xlabel("||f-sym f||")
    # plt.ylabel("Deviation in Angular Momentum")
    degree_potential = 10
    valid_powers_potential = [k for k in product(range(-degree_potential, degree_potential + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree_potential and sum(ki for ki in k)==0]
    theta0 = np.array([2.10, 0.20, -1.15])
    omega0 = np.array([-0.5, 0.25, 0.25])
    dt      = 1e-2          
    n_steps = 1000000      
    test_points = points_circle(100)
    # J_t_unperturbed = verlet_integrate(theta0, omega0, F_perturbed, dt, n_steps, valid_powers_potential, degree_potential, 0)
    psyms = []
    deviation = []
    F_unperturbed = np.apply_along_axis(F, 1, test_points, degree_potential, valid_powers_potential)
    for eps in [3e-3]:
        plt.figure()
        F_p = np.apply_along_axis(F_perturbed, 1, test_points, degree_potential, valid_powers_potential, eps)
        sym = []
        for thetas in test_points:
            sym.append(psym_perturbed(thetas, degree_potential, valid_powers_potential, alpha = 2, epsilon= eps))
        psym = np.sqrt(1/(2*np.pi*100))*np.linalg.norm(F_p-np.array(sym))
        psyms.append(np.linalg.norm(F_p-np.array(sym)))
        J_t_eps = verlet_integrate(theta0, omega0, dt, n_steps, valid_powers_potential, degree_potential, eps)
        plt.ylabel('Total Angular Momentum')
        plt.xlabel('Nb of onesteps')
        plt.axvline(x=round(1/psym/(dt)), color='black', linestyle='--', linewidth=2, label=f"1/||f-symf||")
        plt.plot([i for i in range(n_steps+1)], J_t_eps, label=f"||f-sym(f)||={psym:.2e}")
        plt.legend(loc='upper right')
        plt.show()
        # np.save(f"C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//2dJteps{eps}.npy", J_t_eps)
        # deviation.append(np.max(J_t_unperturbed-J_t_eps))
    # plt.loglog(psyms, deviation, marker='o')
    # plt.show()

if __name__ != "__main__":
    # plt.title("Approximate conservation of Angular momentum ")
    # plt.xlabel("||f-sym f||")
    # plt.ylabel("Deviation in Angular Momentum")
    degree_potential = 10
    valid_powers_potential = [k for k in product(range(-degree_potential, degree_potential + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree_potential and sum(ki for ki in k)==0]
    theta0 = np.array([2.10, 0.20, -1.15])
    omega0 = np.array([-0.5, 0.25, 0.25])
    dt      = 1e-2          
    n_steps = 100000      
    test_points = points_circle(100)
    # J_t_unperturbed = verlet_integrate(theta0, omega0, F_perturbed, dt, n_steps, valid_powers_potential, degree_potential, 0)
    psyms = []
    deviation = []
    F_unperturbed = np.apply_along_axis(F, 1, test_points, degree_potential, valid_powers_potential)
    plt.figure()
    for eps in [1e-2, 1e-3]:
        F_p = np.apply_along_axis(F_perturbed, 1, test_points, degree_potential, valid_powers_potential, eps)
        sym = []
        for thetas in test_points:
            sym.append(psym_perturbed(thetas, degree_potential, valid_powers_potential, alpha = 2, epsilon= eps))
        psym = np.sqrt(1/(2*np.pi*100))*np.linalg.norm(F_p-np.array(sym))
        psyms.append(np.linalg.norm(F_p-np.array(sym)))
        J_t_eps = verlet_integrate(theta0, omega0, dt, n_steps, valid_powers_potential, degree_potential, eps)
        deviation.append(np.max(np.abs(J_t_eps)))
    plt.ylabel('Max(J)')
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel('||f-sym f||')
    plt.plot(psyms, deviation, marker="o",label=f"Maximal deviation in Angular Momentum (Stable Regime), dt=0.01")
    plt.legend(loc='upper right')
    plt.show()
    
    
if __name__ != "__main__":
    # plt.title("Approximate conservation of Angular momentum ")
    # plt.xlabel("||f-sym f||")
    # plt.ylabel("Deviation in Angular Momentum")
    degree_potential = 10
    valid_powers_potential = [k for k in product(range(-degree_potential, degree_potential + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree_potential and sum(ki for ki in k)==0]
    dt      = 5e-2          
    n_steps = 10000000      
    test_points = points_circle(100)
    # J_t_unperturbed = verlet_integrate(theta0, omega0, F_perturbed, dt, n_steps, valid_powers_potential, degree_potential, 0)
    psyms = []
    deviation = []
    F_unperturbed = np.apply_along_axis(F, 1, test_points, degree_potential, valid_powers_potential)
    plt.figure()
    plt.ylabel('Total Angular Momentum')
    plt.xlabel('Nb of onesteps')
    for i in range(1):
        theta0 = points_circle(1)[0]
        omega0 = np.random.uniform(-1,1, 2)
        omega0 = np.append(omega0, -omega0[0]-omega0[1])
        for c,eps in enumerate([1e-2]):
            if c == 0:
                colour = 'blue'
            elif c == 1:
                colour = 'red'
            elif c ==2:
                colour = 'green'
            elif c ==3:
                colour = 'yellow'
            elif c == 4:
                colour = 'orange'

            F_p = np.apply_along_axis(F_perturbed, 1, test_points, degree_potential, valid_powers_potential, eps)
            sym = []
            for thetas in test_points:
                sym.append(psym_perturbed(thetas, degree_potential, valid_powers_potential, alpha = 2, epsilon= eps))
            psym = np.sqrt(1/(2*np.pi*100))*np.linalg.norm(F_p-np.array(sym))
            psyms.insert(1,np.linalg.norm(F_p-np.array(sym)))
            J_t_eps = verlet_integrate(theta0, omega0, dt, n_steps, valid_powers_potential, degree_potential, eps)
            np.save(f'C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//ApproxJ2d//J{eps}_{dt}_{i}.npy', J_t_eps)
            if i == 0:
                plt.plot([i for i in range(n_steps+1)], J_t_eps, color = colour, label=f"||f-sym(f)||={psym:.2e}")
            else:
                plt.plot([i for i in range(n_steps+1)], J_t_eps, color = colour, label='_nolegend_')
    plt.legend(loc='upper right')
    plt.show()
    
    
# fig = plt.figure()
# plt.ylabel('Total Angular Momentum')
# plt.xlabel('Nb of onesteps')
# i=0
# n_steps = 10000000      
# psyms= [2.32e-1,4.63e-2, 4.63e-3, 4.63e-4]
# for c,eps in enumerate([5e-2,1e-2, 1e-3, 1e-4]):
#     J_t_eps = np.load(f'C://Users//zziyu\Desktop//RAMathUBC//ZZY//RAMathUBC//ApproxJ2d//J{eps}_{dt}_{i}.npy')
#     psym = psyms[c]

#     if c == 0:
#         colour = 'blue'
#         plt.plot([i for i in range(n_steps+1)], J_t_eps, color = colour, label=f"||f-sym(f)||={psym:.2e}", alpha=0.2)

#     elif c == 1:
#         colour = 'red'
#         plt.plot([i for i in range(2000000)], J_t_eps[:2000000], color = colour, label=f"||f-sym(f)||={psym:.2e}")
#         plt.plot([i for i in range(2000000, n_steps+1)], J_t_eps[2000000:], color = colour, label=f"__nolabel__", alpha = 0.2)
#     elif c ==2:
#         colour = 'green'
#         plt.plot([i for i in range(n_steps+1)], J_t_eps, color = colour, label=f"||f-sym(f)||={psym:.2e}")

#     elif c ==3:
#         colour = 'yellow'
#         plt.plot([i for i in range(n_steps+1)], J_t_eps, color = colour, label=f"||f-sym(f)||={psym:.2e}")

#     elif c == 4:
#         colour = 'orange'

# plt.legend(loc='lower right')
# plt.show()
    
        # np.save(f"C://Users//zziyu//Desktop//RAMathUBC//ZZY//RAMathUBC//2dJteps{eps}.npy", J_t_eps)
        # deviation.append(np.max(J_t_unperturbed-J_t_eps))
    # plt.loglog(psyms, deviation, marker='o')
    # plt.show()
if __name__ != "__main__":
    # plt.title("Approximate conservation of Angular momentum ")
    # plt.xlabel("||f-sym f||")
    # plt.ylabel("Deviation in Angular Momentum")
    degree_potential = 10
    valid_powers_potential = [k for k in product(range(-degree_potential, degree_potential + 1), repeat=3) if sum(abs(ki) for ki in k) <= degree_potential and sum(ki for ki in k)==0]
    theta0 = np.array([2.10, 0.20, -1.15])
    omega0 = np.array([-0.5, 0.25, 0.25])
    dt      = 5e-2          
    test_points = points_circle(100)
    # J_t_unperturbed = verlet_integrate(theta0, omega0, F_perturbed, dt, n_steps, valid_powers_potential, degree_potential, 0)
    psyms = []
    hitting_times = []
    plt.figure()
    F_unperturbed = np.apply_along_axis(F, 1, test_points, degree_potential, valid_powers_potential)
    for eps in [5e-2, 1e-3, 1e-4]:
        F_p = np.apply_along_axis(F_perturbed, 1, test_points, degree_potential, valid_powers_potential, eps)
        sym = []
        for thetas in test_points:
            sym.append(psym_perturbed(thetas, degree_potential, valid_powers_potential, alpha = 2, epsilon= eps))
        psym = np.sqrt(1/(2*np.pi*100))*np.linalg.norm(F_p-np.array(sym))
        psyms.append(np.linalg.norm(F_p-np.array(sym)))
        hitting_times.append(hitting_time(theta0, omega0, dt, np.sqrt(psym), valid_powers_potential, degree_potential, eps))
        plt.ylabel('Total Angular Momentum')
        plt.xlabel('Nb of onesteps')
        plt.plot(psyms, hitting_times, label=f"||f-sym(f)||={psym:.2e}")
        plt.legend(loc='upper right')
        plt.show()