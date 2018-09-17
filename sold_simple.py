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

SC_EXAMPLE = 9 # 8, 9, 20, 55

# Mesh
NUM_CELL = 33
mesh = UnitSquareMesh(NUM_CELL,NUM_CELL)
h = CellSize(mesh)
cell_volume = CellVolume(mesh)
DG0 = FunctionSpace(mesh, "DG", 0)

# Whole Boundary
def whole_boundary(x, on_boundary):
  return on_boundary

cut_b_elem_dofs = get_boundary(mesh, DG0)

setups = [
	{ "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 3, "W_TYPE": "DG", "W_DEGREE": 1 },
]

global_results = []

for setup in setups:
	# Function Spaces on the mesh
	V = FunctionSpace(mesh, setup["V_TYPE"], setup["V_DEGREE"])
	v = TestFunction(V)
	W = FunctionSpace(mesh, setup["W_TYPE"], setup["W_DEGREE"])
	bc_V_zero = DirichletBC(V, 0., whole_boundary)

	# Data
	bcs, epsilon, c, b, f, u_exact = sc_setup(V, SC_EXAMPLE)
	b_perp = as_vector([( b[1]/sqrt(b[0]**2+b[1]**2)),
		                  (-b[0]/sqrt(b[0]**2+b[1]**2))]) # ! possible division by 0

	# Basic Definitions
	p = 1 # Constant(V.ufl_element().degree())
	tau = compute_tau(W, h, p, epsilon, b)
	uh = solve_supg(V, bcs, epsilon, b, c, f, tau)

	tau2 = compute_sold_tau_codina(uh, 0.7, W, h, epsilon, b, c, f)
	uh = solve_sold(V, bcs, epsilon, b, b_perp, c, f, tau, tau2)

	one = project(1., V)
	area = assemble(one*dx)
	h_average = assemble(h*dx)/area
	
	error_function = Function(V, assemble(abs(uh-u_exact)*v*dx))
	l2_norm_of_error = norm(error_function, 'l2')
	global_result = {'V_dofs': V.dofmap().global_dimension(),
	                 'W_dofs': W.dofmap().global_dimension(),
	                 'phi': 1.,
	                 'phi_30': 1.,
	                 'h': h_average,
	                 'error_l2': l2_norm_of_error}
	global_results.append(global_result)
	results = []
	rs.make_results(SC_EXAMPLE, NUM_CELL, V, W, uh, u_exact, tau, 1., results)

# Global results
rs.make_global_results(SC_EXAMPLE, global_results)
