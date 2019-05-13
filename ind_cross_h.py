from dolfin import *
from scipy.optimize import minimize
import numpy as np
import time as pyt
import pprint
coth = lambda x: 1./np.tanh(x)

from fenicsopt.core.convdif import *
from fenicsopt.examples.sc_examples import sc_setup
import fenicsopt.exports.results as rs

SC_EXAMPLE = 20 # 8, 9, 20, 55

setup = { "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 0 }

global_results = []

for NUM_CELL in range(7, 100, 1):
	# Mesh
	mesh = UnitSquareMesh(NUM_CELL,NUM_CELL)
	h = CellDiameter(mesh)
	n = FacetNormal(mesh)
	cell_volume = CellVolume(mesh)
	DG0 = FunctionSpace(mesh, "DG", 0)
	dg0 = TestFunction(DG0)

	# Whole Boundary
	def whole_boundary(x, on_boundary):
		return on_boundary

	cut_b_elem_dofs = get_boundary(mesh, DG0)

	# Function Spaces on the mesh
	V =  FunctionSpace(mesh, setup["V_TYPE"], setup["V_DEGREE"])
	W =  FunctionSpace(mesh, setup["W_TYPE"], setup["W_DEGREE"])
	bc_V_zero = DirichletBC(V, 0., whole_boundary)

	# Trial And Test Functions
	u   = TrialFunction(V)
	psi = TrialFunction(V)
	v   = TestFunction(V)
	w   = TestFunction(W)

	# Data
	bcs, epsilon, c, b, f, u_exact = sc_setup(V, SC_EXAMPLE)
	b_perp = as_vector([( b[1]/sqrt(b[0]**2+b[1]**2)),
		                  (-b[0]/sqrt(b[0]**2+b[1]**2))]) # ! possible division by 0

	# Basic Definitions
	p = 1 # Constant(V.ufl_element().degree())
	tau = compute_tau(W, h, p, epsilon, b)

	# Phi and dPhi Functions
	def phi(tau):
		global results
		global phi_30
		yh = Function(W)
		yh.vector()[:] = tau
		error = value_of_ind_cross(V, cut_b_elem_dofs, bcs, epsilon, b, b_perp, c, f, yh)
		t_length = pyt.time()-start
		results.append([t_length,error])
		if t_length < 30:
			phi_30 = error
		return error
	def dPhi(tau):
		yh = Function(W)
		yh.vector()[:] = tau
		D_Phi_h = der_of_ind_cross(V, W, cut_b_elem_dofs, bcs, bc_V_zero, epsilon, b, b_perp, c, f, yh)
		der = D_Phi_h.vector().get_local()
		return der

	# Minimization (Bounds Are Set Up First)
	initial = tau.vector().get_local()
	lower_bound = 0 * initial
	upper_bound = 10 * initial
	yh_bounds = np.array([lower_bound,upper_bound])
	yh_bounds = np.transpose(yh_bounds)
	results = []
	start = pyt.time()
	phi_30 = 1e+10
	res = minimize(phi, initial, method='L-BFGS-B', jac=dPhi, bounds=yh_bounds,
	  options={'gtol': 1e-14, 'ftol': 1e-14, 'disp': True})

	# Results Of Minimization
	yh = Function(W)
	yh.vector()[:] = res.x
	uh = solve_supg(V, bcs, epsilon, b, c, f, yh)
	res_phi = phi(yh.vector())
	one = project(1., V)
	area = assemble(one*dx)
	h_average = assemble(h*dx)/area

	error_function = Function(V, assemble(abs(uh-u_exact)*v*dx))
	norm_of_error = norm(error_function, 'h1')
	global_result = {'V_dofs': V.dofmap().global_dimension(),
	                 'W_dofs': W.dofmap().global_dimension(),
	                 'phi': res_phi,
	                 'phi_30': phi_30,
	                 'h': h_average,
	                 'error_l2': norm_of_error}
	global_results.append(global_result)
	rs.make_results_h('RESULTS/' + str(SC_EXAMPLE) + 'indCrossH', NUM_CELL, V, W, uh, u_exact, yh, res_phi)

rs.make_global_results_h('RESULTS/' + str(SC_EXAMPLE) + 'indCrossH', global_results)

