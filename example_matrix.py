import numpy as np
import sympy
from scipy.linalg import solve
import math

def mod_matrix(matrix, field_size):
	for i in range(0, len(matrix)):
		for j in range(0, len(matrix[i])):
			matrix[i][j] = math.floor(matrix[i][j]) % field_size

def reduce_matrix(matrix):
	_, inds = sympy.Matrix(matrix.T).T.rref()
	cols_to_delete = []
	for i in range(0, len(matrix[0])):
		if i not in inds:
			cols_to_delete.append(i)

	for i in cols_to_delete:
		matrix = np.delete(matrix, i, 1)
	return matrix

x = np.array([[1, 3, 2], [2, 2, 1], [3, 1, 3]])
gamma = np.array([[1, 1, 2, 3], [1, 3, 2, 1], [1, 1, 3, 2]])
gamma = reduce_matrix(gamma)

p = np.dot(x, gamma)
#mod_matrix(p, 5)
p = reduce_matrix(p)
print(gamma)
print(p)
print(x)
x_approx = np.dot(p, np.linalg.inv(gamma))
mod_matrix(x_approx, 5)
print(x_approx)
