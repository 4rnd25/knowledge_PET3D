"""
Created on 13 05 2024 08:46

@author: ISAC - pettirsch
"""
import numpy as np

def allign_normal_vector_for_hnf(normal_vector, plane_point):

    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector