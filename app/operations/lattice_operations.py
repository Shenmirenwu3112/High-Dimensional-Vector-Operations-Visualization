### lattice_operations.py
import numpy as np
import scipy.interpolate

np.seterr(divide='ignore', invalid='ignore')

def gram_schmidt(Basis):
    """
    Performs Gram-Schmidt orthogonalization on the basis.

    Parameters:
        Basis (np.ndarray): The basis matrix where each row is a basis vector.

    Returns:
        U (np.ndarray): The orthogonalized basis.
    """
    n_rows = Basis.shape[0]
    U = np.zeros_like(Basis, dtype=np.float64)
    U[0] = Basis[0]
    for i in range(1, n_rows):
        proj = sum((np.dot(Basis[i], U[j]) / np.dot(U[j], U[j])) * U[j] for j in range(i))
        U[i] = Basis[i] - proj
    return U

def compute_coeff(Basis, U, mu):
    """
    Computes the coefficients mu for the LLL algorithm.

    Parameters:
        Basis (np.ndarray): The basis matrix.
        U (np.ndarray): The orthogonalized basis vectors.
        mu (np.ndarray): The array to store computed mu values.

    Returns:
        mu (np.ndarray): Updated mu array with computed coefficients.
    """
    n_rows = Basis.shape[0]
    for i in range(n_rows):
        for j in range(i):
            mu[i][j] = np.dot(Basis[i], U[j]) / np.dot(U[j], U[j])
    return mu

def LLL_reduction(Basis, delta=0.99):
    """
    Performs LLL lattice reduction on the Basis.

    Parameters:
        Basis (np.ndarray): The basis matrix where each row is a basis vector.
        delta (float): Lovász condition parameter. Typically between 0.5 and 1.0.

    Returns:
        Basis (np.ndarray): The reduced basis matrix.
    """
    n_rows, n_cols = Basis.shape
    U = gram_schmidt(Basis)
    mu = np.zeros((n_rows, n_cols))
    k = 1

    while k < n_rows:
        # Size reduction
        for j in range(k - 1, -1, -1):
            mu = compute_coeff(Basis, U, mu)
            if abs(mu[k][j]) > 0.5:
                Basis[k] = Basis[k] -  np.round(mu[k][j]) * Basis[j]
                U = gram_schmidt(Basis)

        # Lovász condition
        mu = compute_coeff(Basis, U, mu)
        mu_k_k1 = mu[k][k - 1]
        norm_Uk = np.dot(U[k], U[k])
        norm_Uk1 = np.dot(U[k - 1], U[k - 1])
        if norm_Uk >= (delta - mu_k_k1 ** 2) * norm_Uk1:
            k += 1
        else:
            Basis[[k, k - 1]] = Basis[[k - 1, k]]  # Swap rows
            U = gram_schmidt(Basis)
            k = max(k - 1, 1)
    return Basis
