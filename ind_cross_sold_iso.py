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

'''
setups = [
	{ "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 0 },
	{ "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 2 },

	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 0 },
	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 2 },
	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 3 },

	{ "V_TYPE": "CG", "V_DEGREE": 3, "W_TYPE": "DG", "W_DEGREE": 0 },
	{ "V_TYPE": "CG", "V_DEGREE": 3, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 3, "W_TYPE": "DG", "W_DEGREE": 2 },
	{ "V_TYPE": "CG", "V_DEGREE": 3, "W_TYPE": "DG", "W_DEGREE": 3 },
	{ "V_TYPE": "CG", "V_DEGREE": 3, "W_TYPE": "DG", "W_DEGREE": 4 },

	{ "V_TYPE": "CG", "V_DEGREE": 4, "W_TYPE": "DG", "W_DEGREE": 0 },
	{ "V_TYPE": "CG", "V_DEGREE": 4, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 4, "W_TYPE": "DG", "W_DEGREE": 2 },
	{ "V_TYPE": "CG", "V_DEGREE": 4, "W_TYPE": "DG", "W_DEGREE": 3 },
	{ "V_TYPE": "CG", "V_DEGREE": 4, "W_TYPE": "DG", "W_DEGREE": 4 },
	{ "V_TYPE": "CG", "V_DEGREE": 4, "W_TYPE": "DG", "W_DEGREE": 5 },

	{ "V_TYPE": "CG", "V_DEGREE": 5, "W_TYPE": "DG", "W_DEGREE": 0 },
	{ "V_TYPE": "CG", "V_DEGREE": 5, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 5, "W_TYPE": "DG", "W_DEGREE": 2 },
	{ "V_TYPE": "CG", "V_DEGREE": 5, "W_TYPE": "DG", "W_DEGREE": 3 },
	{ "V_TYPE": "CG", "V_DEGREE": 5, "W_TYPE": "DG", "W_DEGREE": 4 },
	{ "V_TYPE": "CG", "V_DEGREE": 5, "W_TYPE": "DG", "W_DEGREE": 5 },
]
'''
setups = [
	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 1 },
]

global_results = []

for setup in setups:
	# Function Spaces on the mesh
	V = FunctionSpace(mesh, setup["V_TYPE"], setup["V_DEGREE"])
	v = TestFunction(V)
	W = FunctionSpace(mesh, setup["W_TYPE"], setup["W_DEGREE"])
	# Function space for the convection field
	B = VectorFunctionSpace(mesh, "DG", setup["V_DEGREE"]-1)
	# Other spaces for precise numerical integration
	V_grad_norm = FunctionSpace(mesh, "DG", (setup["V_DEGREE"]-1))
	V_grad_squared = FunctionSpace(mesh, "DG", 2*(setup["V_DEGREE"]-1))
	
	# Dirichlet Boundary Condition
	bc_V_zero = DirichletBC(V, 0., whole_boundary)

	# Data
	bcs, epsilon, c, b, f, u_exact = sc_setup(V, SC_EXAMPLE)
	b_perp = as_vector([( b[1]/sqrt(b[0]**2+b[1]**2)),
		                  (-b[0]/sqrt(b[0]**2+b[1]**2))]) # ! possible division by 0
	b = project(b, B)
	b_perp = project(b_perp, B)
	
	# Basic Definitions
	p = 1 # Constant(V.ufl_element().degree())
	tau = compute_tau(W, h, p, epsilon, b)
	uh = solve_supg(V, bcs, epsilon, b, c, f, tau)
	
	b_parallel = compute_sold_iso_b_parallel(mesh, V, B, uh, b)
	
	sigma = compute_sold_iso_sigma(mesh, V, B, W, uh, 1., h, b, b_parallel)
	print residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma)
	
	uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma)
	sigma = compute_sold_iso_sigma(mesh, V, B, W, uh, 1., h, b, b_parallel)
	print residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma)
	
	uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma)
	sigma = compute_sold_iso_sigma(mesh, V, B, W, uh, 1., h, b, b_parallel)
	print residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma)
	
	uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma)
	sigma = compute_sold_iso_sigma(mesh, V, B, W, uh, 1., h, b, b_parallel)
	print residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma)
	
	uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma)
	sigma = compute_sold_iso_sigma(mesh, V, B, W, uh, 1., h, b, b_parallel)
	print residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma)
	
	uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma)
	print residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma)
	sigma = compute_sold_iso_sigma(mesh, V, B, W, uh, 1., h, b, b_parallel)
	print residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma)
	
	
	plot(uh)
	interactive()


	# Phi and dPhi Functions
	def phi(tau):
		global results
		global phi_30
		tau1 = tau[:len(tau)/2]
		sigma = tau[len(tau)/2:]
		yh = Function(W)
		yh.vector()[:] = tau1
		yh2 = Function(W)
		yh2.vector()[:] = sigma
		error = value_of_ind_cross_sold_iso(V, cut_b_elem_dofs, bcs,
			epsilon, b, b_perp, b_parallel, c, f, yh, yh2)
		t_length = pyt.time()-start
		results.append([t_length,error])
		if t_length < 30:
			phi_30 = error
		return error

	def dPhi(tau):
		tau1 = tau[:len(tau)/2]
		sigma = tau[len(tau)/2:]
		yh = Function(W)
		yh.vector()[:] = tau1
		yh2 = Function(W)
		yh2.vector()[:] = sigma
		D_Phi_h_supg, D_Phi_h_sold = der_of_ind_cross_sold_iso(V, W,
			cut_b_elem_dofs, bcs, bc_V_zero,
			epsilon, b, b_perp, b_parallel, c, f, yh, yh2)
		der1 = D_Phi_h_supg.vector().array()
		der2 = D_Phi_h_sold.vector().array()
		der = np.concatenate((der1, der2), axis=1)
		return der

	# Minimization (Bounds Are Set Up First)
	initial1 = tau.vector().array()
	initial2 = 0.2 * sigma.vector().array()
	initial = np.concatenate((initial1, initial2), axis=1)
	lower_bound1 = 0 * initial1
	upper_bound1 = 1 * initial1
	lower_bound2 = 0 * initial2
	upper_bound2 = 1.5 * initial2
	yh_bounds1 = np.array([lower_bound1,upper_bound1])
	yh_bounds2 = np.array([lower_bound2,upper_bound2])
	yh_bounds = np.concatenate((yh_bounds1, yh_bounds2), axis=1)
	print(yh_bounds)
	yh_bounds = np.transpose(yh_bounds)

	results = []
	start = pyt.time()
	phi_30 = 1e+10
	res = minimize(phi, initial, method='L-BFGS-B', jac=dPhi, bounds=yh_bounds,
	  options={'gtol': 1e-14, 'ftol': 1e-14, 'maxiter': 30, 'disp': True})

	# Results Of Minimization
	yh1 = Function(W)
	yh2 = Function(W)
	tau = res.x
	tau1 = tau[:len(tau)/2]
	sigma = tau[len(tau)/2:]
	yh1.vector()[:] = tau1
	yh2.vector()[:] = sigma
	
	plot(yh2)
	
	uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, yh1, yh2)
	res_phi = phi(tau)
	
	one = project(1., V)
	area = assemble(one*dx)
	h_average = assemble(h*dx)/area
	
	error_function = Function(V, assemble(abs(uh-u_exact)*v*dx))
	l2_norm_of_error = norm(error_function, 'l2')
	global_result = {'V_dofs': V.dofmap().global_dimension(),
	                 'W_dofs': W.dofmap().global_dimension(),
	                 'phi': res_phi,
	                 'phi_30': phi_30,
	                 'h': h_average,
	                 'error_l2': l2_norm_of_error}
	global_results.append(global_result)
	rs.make_results(SC_EXAMPLE, NUM_CELL, V, W, uh, u_exact, yh1, res_phi, results)
	rs.make_results_sold_par(SC_EXAMPLE, NUM_CELL, V, W, yh2)

# Global results
rs.make_global_results(SC_EXAMPLE, global_results)

