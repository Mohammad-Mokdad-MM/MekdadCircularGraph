import numpy as np


def create_permutation_matrix(v):
    v = np.asarray(v)
    n = len(v)
    m = int(np.max(v))
    M = np.zeros((n, m))
    for i in range(n):
        M[i, int(v[i]) - 1] = 1
    return M


def filter_matrix_by_threshold(matrix, threshold):
    filtered = matrix.copy()
    filtered[np.abs(filtered) <= threshold] = 0
    return filtered


def reorder_matrix(matrix, labelling):
    valid_indices = labelling > 0
    filtered_matrix = matrix[valid_indices][:, valid_indices]
    labelling_short = labelling[valid_indices]
    return filtered_matrix, labelling_short


def load_brain_connectivity_data(matrix_file, labelling_file, names_file, colormap_file):
    import pandas as pd
    matrix = pd.read_csv(matrix_file, index_col=0).values
    labelling = pd.read_csv(labelling_file, header=None).values.flatten()
    names = pd.read_csv(names_file, header=None).values.flatten()
    colormap = pd.read_csv(colormap_file, header=None, sep=';').values
    return {
        'matrix': matrix,
        'labelling': labelling,
        'names': names,
        'colormap': colormap,
    }


def validate_adjacency_matrix(matrix):
    errors = []
    if matrix.ndim != 2:
        errors.append(f"Matrix must be 2D, got {matrix.ndim}D")
        return False, errors
    if matrix.shape[0] != matrix.shape[1]:
        errors.append(f"Matrix must be square, got shape {matrix.shape}")
        return False, errors
    if not np.allclose(matrix, matrix.T, rtol=1e-5):
        errors.append("Matrix is not symmetric")
    if np.any(np.isnan(matrix)):
        errors.append("Matrix contains NaN values")
    if np.any(np.isinf(matrix)):
        errors.append("Matrix contains infinite values")
    return len(errors) == 0, errors


if __name__ == "__main__":
    v = np.array([2, 1, 3, 4])
    M = create_permutation_matrix(v)
    print("Test permutation matrix:")
    print(f"Input vector: {v}")
    print(f"Permutation matrix:\n{M}")
    print(f"Matrix shape: {M.shape}")