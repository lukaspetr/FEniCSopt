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

SC_EXAMPLE = 55 # 8, 9, 20, 55

# Mesh
NUM_CELL = 40
mesh = UnitSquareMesh(NUM_CELL,NUM_CELL)
h = CellDiameter(mesh)
cell_volume = CellVolume(mesh)
DG0 = FunctionSpace(mesh, "DG", 0)

# Whole Boundary
def whole_boundary(x, on_boundary):
  return on_boundary

cut_b_elem_dofs = get_boundary(mesh, DG0)

setups = [
	{ "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 0 },
	{ "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 0 },
	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 1 },
	{ "V_TYPE": "CG", "V_DEGREE": 2, "W_TYPE": "DG", "W_DEGREE": 2 },
]

global_results = []

for setup in setups:
	# Function Spaces on the mesh
	V =  FunctionSpace(mesh, setup["V_TYPE"], setup["V_DEGREE"])
	v   = TestFunction(V)
	W =  FunctionSpace(mesh, setup["W_TYPE"], setup["W_DEGREE"])
	bc_V_zero = DirichletBC(V, 0., whole_boundary)

	# Data
	bcs, epsilon, c, b, f, u_exact = sc_setup(V, SC_EXAMPLE)
	b_perp = as_vector([( b[1]/sqrt(b[0]**2+b[1]**2)),
		                  (-b[0]/sqrt(b[0]**2+b[1]**2))]) # ! possible division by 0

	# Basic Definitions
	p = 1 # Constant(V.ufl_element().degree())
	tau = compute_tau(W, h, p, epsilon, b)
	uh = solve_supg(V, bcs, epsilon, b, c, f, tau)

	# Phi and dPhi Functions
	def phi(tau):
		global results
		global phi_30
		yh = Function(W)
		yh.vector()[:] = tau
		error = value_of_ind_cross(V, cut_b_elem_dofs, bcs,
			epsilon, b, b_perp, c, f, yh)
		t_length = pyt.time()-start
		results.append([t_length,error])
		if t_length < 30:
			phi_30 = error
		return error

	def dPhi(tau):
		yh = Function(W)
		yh.vector()[:] = tau
		D_Phi_h_supg = der_of_ind_cross(V, W,
			cut_b_elem_dofs, bcs, bc_V_zero,
			epsilon, b, b_perp, c, f, yh)
		der = D_Phi_h_supg.vector().get_local()
		return der

	# Minimization (Bounds Are Set Up First)
	initial = tau.vector().get_local()
	
	print(initial)
	
	lower_bound = 0 * initial
	upper_bound = 5 * initial
	yh_bounds = np.array([lower_bound,upper_bound])
	print(yh_bounds)
	yh_bounds = np.transpose(yh_bounds)

	results = []
	start = pyt.time()
	phi_30 = 1e+10
	res = minimize(phi, initial, method='L-BFGS-B', jac=dPhi, bounds=yh_bounds,
	  options={'gtol': 1e-14, 'ftol': 1e-15, 'maxiter': 350, 'disp': True})

	# Results Of Minimization
	yh = Function(W)
	tau = res.x
	yh.vector()[:] = tau
	uh = solve_supg(V, bcs, epsilon, b, c, f, yh)
	res_phi = phi(tau)
	
	
	"""
	# Line evaluation after the last iteration (if any)
	direction = dPhi(tau)
	step = 0.1
	begin = -1.
	end = 10.
	current = begin
	data = []
	while current <= end:
	  current_tau = np.add(tau,current*direction)
	  current_phi = phi(current_tau)
	  point = {'position': current,
	           'phi': current_phi}
	  data.append(point)
	  current += step
	
	rs.make_line_results(SC_EXAMPLE, data)
	
	pprint.pprint(data)
	"""
	
	# Second version of line evaluation after the last iteration (if any) - from value 0
	steps = 800
	begin = np.zeros(len(tau))
	end = tau
	basic_line = np.subtract(end, begin)
	length = 40.
	current = begin
	data = []
	step = 3
	while step <= steps:
	  current_pos = float(step)/steps * length
	  current_tau = np.add(begin,current_pos*basic_line)
	  current_phi = phi(current_tau)
	  point = {'position': current_pos,
	           'phi': current_phi}
	  data.append(point)
	  step += 1
	
	rs.make_line_results('RESULTS/' + str(SC_EXAMPLE) + 'indCrossDirectionSDFEM', V, W, data)
	
	pprint.pprint(data)
	
	
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
	rs.make_results('RESULTS/' + str(SC_EXAMPLE) + 'indCrossDirectionSDFEM', NUM_CELL, V, W, uh, u_exact, yh, res_phi, results)

# Global results
rs.make_global_results('RESULTS/' + str(SC_EXAMPLE) + 'indCrossDirectionSDFEM', global_results)
