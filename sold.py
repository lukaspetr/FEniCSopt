from dolfin import *
from scipy.optimize import minimize
import numpy as np
import time as pyt
import pprint
coth = lambda x: 1./np.tanh(x)

from fenicsopt.core.convdif import *
from fenicsopt.examples.sc_examples import sc_setup
import fenicsopt.exports.results as rs

################################################################################

SC_EXAMPLE = 1 # 8, 9, 20, 55

# Mesh
NUM_CELL = 24
mesh = UnitSquareMesh(NUM_CELL,NUM_CELL)
h = CellDiameter(mesh)
cell_volume = CellVolume(mesh)
DG0 = FunctionSpace(mesh, "DG", 0)

# Whole Boundary
def whole_boundary(x, on_boundary):
  return on_boundary

cut_b_elem_dofs = get_boundary(mesh, DG0)

setup = { "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 0 }

# Function Spaces on the mesh
V = FunctionSpace(mesh, setup["V_TYPE"], setup["V_DEGREE"])
v = TestFunction(V)
W = FunctionSpace(mesh, setup["W_TYPE"], setup["W_DEGREE"])
bc_V_zero = DirichletBC(V, 0., whole_boundary)

# Data
bcs, epsilon, c, b, f, u_exact = sc_setup(V, SC_EXAMPLE)
epsilon = 1e-4
b_perp = as_vector([( b[1]/sqrt(b[0]**2+b[1]**2)),
                    (-b[0]/sqrt(b[0]**2+b[1]**2))]) # ! possible division by 0

# Basic Definitions
p = 1 # Constant(V.ufl_element().degree())
tau = compute_tau(W, h, p, epsilon, b)

uh = solve_supg(V, bcs, epsilon, b, c, f, tau)
tau2 = iterate_sold_cross(mesh, h, V, W, bcs, epsilon, b, b_perp, c, f, tau, uh, 0.9999)
uh = solve_sold_cross(V, bcs, epsilon, b, b_perp, c, f, tau, tau2)

one = project(1., V)
area = assemble(one*dx)
h_average = assemble(h*dx)/area

error_function = Function(V, assemble(abs(uh-u_exact)*v*dx))
l2_norm_of_error = norm(error_function, 'l2')

plot(uh)

results = []
rs.make_results('RESULTS/' + str(SC_EXAMPLE) + 'SOLD', NUM_CELL, V, W, uh, u_exact, tau2, 1., results)
