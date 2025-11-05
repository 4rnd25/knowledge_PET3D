"""
Created on Mai 03 09:47

@author: ISAC - pettirsch
"""
import numpy as np

def gram_schmidt(vectors):
    num_vecs = len(vectors)
    ortho_basis = np.zeros_like(vectors)

    for i in range(num_vecs):
        # Orthogonalization step
        ortho_basis[i] = vectors[i]
        for j in range(i):
            ortho_basis[i] -= np.dot(vectors[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j]) * ortho_basis[
                j]

        # Normalization step
        ortho_basis[i] /= np.linalg.norm(ortho_basis[i])

    return ortho_basis


def modified_gram_schmidt(vectors):
    num_vecs = len(vectors)
    ortho_basis = np.zeros_like(vectors)

    for i in range(num_vecs):
        # Orthogonalization step
        ortho_basis[i] = vectors[i]
        for j in range(i):
            ortho_basis[i] -= np.dot(vectors[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j]) * ortho_basis[j]

        # Normalize the orthogonalized vector
        ortho_basis[i] /= np.linalg.norm(ortho_basis[i])

    # Adjust the orthogonalized vectors to be as close as possible to the original vectors
    rotation_axis = np.cross(ortho_basis[0], ortho_basis[1])
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    dot_prod = np.dot(vectors[1], ortho_basis[1])
    v_1_mag = np.linalg.norm(vectors[1])
    v_2_mag = np.linalg.norm(ortho_basis[1])

    cos_angle = dot_prod / (v_1_mag * v_2_mag)
    angle = np.arccos(cos_angle)

    # Force angle to be positive between 0 and pi
    if angle < 0:
        angle = -angle
        rotation_axis = -rotation_axis


    rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, 2*np.pi-angle/2)

    ortho_basis[1] = np.dot(rotation_matrix, ortho_basis[1])
    ortho_basis[0] = np.dot(rotation_matrix, ortho_basis[0])

    return ortho_basis

def rotation_matrix_from_axis_angle(axis, angle):
    axis /= np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])