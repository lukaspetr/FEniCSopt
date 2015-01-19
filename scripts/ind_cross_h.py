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

# Make properly the function which cuts the dofs of boundary elements
def get_boundary(mesh, DG0):
	bdry_facets = FacetFunction('bool', mesh, False)
	DomainBoundary().mark(bdry_facets, True) # DomainBoundary() is a built-in func
	dg0_dofmap = DG0.dofmap()
	bdry_dofs = np.zeros(1, dtype=np.int32)
	""" only cells with boundary facet
	for cell in cells(mesh):
		if any(f.exterior() for f in facets(cell)):
			bdry_dofs = np.append(bdry_dofs, [dg0_dofmap.cell_dofs(cell.index())])
	"""
	for cell in cells(mesh):
		for myVertex in vertices(cell):
			if any(f.exterior() for f in facets(myVertex)):
			  bdry_dofs = np.append(bdry_dofs, [dg0_dofmap.cell_dofs(cell.index())])
	bdry_dofs.sort()
	cut_b_elem_dofs = Function(DG0)
	cut_b_elem_dofs = project(1., DG0)
	values = cut_b_elem_dofs.vector().get_local()
	values[bdry_dofs] = 0
	cut_b_elem_dofs.vector().set_local(values) # plot(cut_b_elem_dofs)
	return cut_b_elem_dofs

################################################################################

# Compute the Tau Parameter From SUPG Method
def compute_tau(W, h, p, epsilon, b):
	bb = project(dot(b,b), W)
	# Tau Definition (Peclet Number And The Function Of The Peclet Number Are Used)
	peclet = project(bb**0.5*h/(2.*p*epsilon), W)
	# Avoiding possible problems at machine precision level, e.g. in example 9 at [0, 0]
	peclet_array = np.absolute(peclet.vector().array())
	function_of_peclet = np.subtract(coth(peclet_array),np.power(peclet_array,-1))
	f_peclet = Function(W)
	f_peclet.vector()[:] = function_of_peclet
	tau = project((h/(2.*p*conditional(lt(bb,1),1,bb**0.5))*f_peclet), W)
	return tau

################################################################################

# SUPG method to solve convection-diffusion-reaction equation
def solve_conv_diff(V, bcs, epsilon, b, c, f, tau):
  u = TrialFunction(V)
  v = TestFunction(V)
  a = (epsilon*dot(grad(u),grad(v)) + v*dot(b,grad(u)) + c*u*v)*dx +\
      inner(-epsilon*div(grad(u))+dot(b,grad(u))+c*u,tau*dot(b,grad(v)))*dx
  L = f*v*dx + inner(f,tau*dot(b,grad(v)))*dx
  uh = Function(V)
  solve(a == L, uh, bcs)
  return uh

################################################################################

# Error Indicator With Added Crosswind Derivative Control Term
def value_of_ind_cross(V, bcs, epsilon, b, b_perp, c, f, tau):
	fcn_in_ind = lambda u:conditional(gt(u,1), sqrt(u), 2.5*u**2 - 1.5*u**3)
	uh = Function(V)
	uh = solve_conv_diff(V, bcs, epsilon, b, c, f, tau)
	# Indicator
	indicator_assemble = assemble(
	  ((-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f)**2
	    + 0.5 * fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-15)
	  )*dg0*cut_b_elem_dofs*dx)
	indicatorFunction = Function(DG0, indicator_assemble)
	error = assemble(indicatorFunction*dx)
	'''
	error = sum(i for i in indicator_assemble)
	'''
	return error

def der_of_ind_cross(V, bcs, epsilon, b, b_perp, c, f, tau):
	der_of_fcn_in_ind = lambda u:conditional(gt(u,1),0.5/sqrt(u),5.*u-4.5*u**2)
	uh = solve_conv_diff(V, bcs, epsilon, b, c, f, tau)
	# Derivatives
	derivatives_assemble = assemble((2*inner(
	  (-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f),
	  (-epsilon*div(grad(v ))+dot(b,grad(v ))+c*v))
	    + 0.5 * der_of_fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-15)
	                           * sign(dot(b_perp,grad(uh))) * dot(b_perp,grad(v))
	  )*cut_b_elem_dofs*dx)
	derivatives = Function(V, derivatives_assemble)
	# Adjoint Approach To Compute Derivatives According To tau
	a = (epsilon*dot(grad(v),grad(psi)) + psi*dot(b,grad(v)) + c*v*psi)*dx +\
	    inner(-epsilon*div(grad(v))+dot(b,grad(v))+c*v,tau*dot(b,grad(psi)))*dx
	L = derivatives*v*dx
	psih = Function(V)
	solve(a == L, psih, bc_V_zero)
	# Compute D_Phi_h
	D_Phi_h_ufl = (-inner(-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh,
	  w*dot(b,grad(psih))) + inner(f,w*dot(b,grad(psih))))*dx
	D_Phi_h_assemble = assemble(D_Phi_h_ufl)
	D_Phi_h = Function(W, D_Phi_h_assemble)
	return D_Phi_h

################################################################################
################################################################################

SC_EXAMPLE = 20 # 8, 9, 20, 55

setup = { "V_TYPE": "CG", "V_DEGREE": 1, "W_TYPE": "DG", "W_DEGREE": 0 }

global_results = []

for NUM_CELL in range(7, 100, 1):
	# Mesh
	mesh = UnitSquareMesh(NUM_CELL,NUM_CELL)
	h = CellSize(mesh)
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
		error = value_of_ind_cross(V, bcs, epsilon, b, b_perp, c, f, yh)
		t_length = pyt.time()-start
		results.append([t_length,error])
		if t_length < 30:
			phi_30 = error
		return error
	def dPhi(tau):
		yh = Function(W)
		yh.vector()[:] = tau
		D_Phi_h = der_of_ind_cross(V, bcs, epsilon, b, b_perp, c, f, yh)
		der = D_Phi_h.vector().array()
		return der

	# Minimization (Bounds Are Set Up First)
	initial = tau.vector().array()
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
	uh = solve_conv_diff(V, bcs, epsilon, b, c, f, yh)
	res_phi = phi(yh.vector())
	one = project(1., V)
	area = assemble(one*dx)
	h_average = assemble(h*dx)/area
	error_function = Function(V, assemble(abs(uh-u_exact)*v*dx))
	l2_norm_of_error = norm(error_function, 'h1')
	global_result = {'V_dofs': V.dofmap().global_dimension(),
	                 'W_dofs': W.dofmap().global_dimension(),
	                 'phi': res_phi,
	                 'phi_30': phi_30,
	                 'h': h_average,
	                 'error_l2': l2_norm_of_error}
	global_results.append(global_result)
	rs.make_results_h(SC_EXAMPLE, NUM_CELL, V, W, uh, u_exact, yh, res_phi)

rs.make_global_results_h(SC_EXAMPLE, global_results)
