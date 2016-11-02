from fem.definitions import local_stiffness_matrix, local_mass_matrix
from fem.construction import construct_global_stiffness_matrix, construct_global_mass_matrix

from obj.parser import parse_obj
from obj.transforms import to_numpy_array, enforce_fem_order

from math import sin, pi

import logging
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


logging.basicConfig(format="%(name)s :: [%(levelname)s]: %(message)s", level=logging.INFO)

# Load the mesh that will represent our domain
obj = parse_obj('meshes/plane_2d_n4.obj')
obj = to_numpy_array(obj)
obj = enforce_fem_order(obj)

# Set the parameters which will characterise the system
parameters = (1, [0,0], 0)

# Define which the unknown nodes are in the problem.
def known_nodes(v):
    return abs(v[0]) == 1 or abs(v[1]) == 1 or v[0] == 0 or v[1] == 0

# Create the domain
domain = {'obj': obj, 'known_nodes': known_nodes}

# Construct the global matrices for the system.
A = construct_global_stiffness_matrix(domain, parameters)
M = construct_global_mass_matrix(domain, parameters)

# Define f
def f(x, y):
    return (2*pi**2) * (sin(pi * x)) * (sin(pi * y))

# Discreteise our f
verts = [(x,y) for x,y,_ in obj['vertices']]
F = np.array([f(x, y) for x, y in verts])

# Solve the system
u = np.linalg.solve(A.A, M*F)
u = np.append(u, [0.0 for _ in range(44)])

# Plot the result
xs = [x for x, y in verts]
ys = [y for x, y in verts]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, u, c='r', marker='o')
plt.savefig('result.png')
