### vector_operations.py
import numpy as np
import scipy.interpolate

np.seterr(divide='ignore', invalid='ignore')

def interpolate(X, Y, scale=100, method='cubicspline', ret='XY'):
    """
    Interpolates the given data points using the specified method.

    Parameters:
        X (array-like): The x-coordinates of the data points.
        Y (array-like): The y-coordinates of the data points.
        scale (int): The number of points to interpolate between min and max of X.
        method (str): The interpolation method to use. Currently supports 'cubicspline'.
        ret (str): Return type. 'Y' for interpolated Y values, 'XY' for both X and Y.

    Returns:
        If ret == 'Y': Returns the interpolated Y values.
        If ret == 'XY': Returns a tuple (X_plot, Y_plot).
    """
    x_data = np.asarray(X)
    y_data = np.asarray(Y)
    x_plot = np.linspace(x_data.min(), x_data.max(), scale)

    try:
        if method == 'cubicspline':
            spline = scipy.interpolate.CubicSpline(x_data, y_data)
            y_plot = spline(x_plot)
        else:
            raise ValueError("Unsupported interpolation method. Use 'cubicspline'.")

        # Replace non-finite values with zeros
        if np.any(~np.isfinite(y_plot)):
            print("Non-finite values detected in interpolation results. Replacing with zeros.")
            y_plot = np.nan_to_num(y_plot, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception as e:
        print(f"Interpolation failed: {e}")
        y_plot = np.zeros_like(x_plot)

    if ret == 'Y':
        return y_plot
    return x_plot, y_plot

def calculate_angles_with_complementary(Basis):
    """
    Calculates angles between each basis vector and its complementary subspace.

    Parameters:
        Basis (np.ndarray): The basis matrix where each row is a basis vector.

    Returns:
        angles (np.ndarray): Array of angles in degrees between each basis vector and its complementary subspace.
    """
    normal_vectors = calculateNormal(Basis)
    angles = calculateAngle(Basis, normal_vectors)
    return np.abs(angles - 90)

def calculateNormal(Basis):
    """
    Calculates normal vectors for each basis vector with respect to its complementary subspace.

    Parameters:
        Basis (np.ndarray): The basis matrix where each row is a basis vector.

    Returns:
        normal_vectors (np.ndarray): Array of normal vectors corresponding to each basis vector.
    """
    b = Basis.copy()
    # print(b)
    normal_vectors = []
    n_vectors = Basis.shape[0]
    dimension = Basis.shape[1]
    for i in range(n_vectors):
        n_vector = np.zeros(dimension)  # Initialize normal vector
        for j in range(dimension):
            # Create a submatrix by deleting current row and current column
            sub_matrix = np.delete(np.delete(b, i, axis=0), j, axis=1)
            # print(sub_matrix)
            n_vector[j] = (-1) ** (j + i) * np.linalg.det(sub_matrix)
        n_vector = np.round(n_vector)
        normal_vectors.append(n_vector)
    return np.array(normal_vectors)

def calculateAngle(Basis, normal_vectors):
    """
    Calculates the angles between each basis vector and its corresponding normal vector.

    Parameters:
        Basis (np.ndarray): The basis matrix where each row is a basis vector.
        normal_vectors (np.ndarray): Array of normal vectors corresponding to each basis vector.

    Returns:
        angles (np.ndarray): Array of angles in degrees.
    """
    angles = []
    for i in range(Basis.shape[0]):
        b = Basis[i]
        n = normal_vectors[i]
        norm_b = np.linalg.norm(b)
        norm_n = np.linalg.norm(n)
        if norm_b == 0 or norm_n == 0:
            print(f"Zero norm encountered for basis vector {i}. Setting angle to 0.")
            angles.append(0.0)
            continue
        cos_theta = np.dot(b, n) / (norm_b * norm_n)
        # Ensure the cosine value is within valid range to avoid numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        angles.append(angle)
    return np.round(np.array(angles))
