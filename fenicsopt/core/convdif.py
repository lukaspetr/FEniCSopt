from dolfin import *
import numpy as np
coth = lambda x: 1./np.tanh(x)
eta = lambda x: 2.*x*(1-x)

# Make properly the function which cuts the dofs of boundary elements
def get_boundary(mesh, DG0):
	bdry_facets = FacetFunction('bool', mesh, False)
	DomainBoundary().mark(bdry_facets, True) # DomainBoundary() is a built-in func
	dg0_dofmap = DG0.dofmap()
	bdry_dofs = np.zeros(1, dtype=np.int32)
	"""
	# only cells with boundary facet
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
	# Tau Definition (Peclet Number And The Function Of The Pecl. Number Are Used)
	peclet = project(bb**0.5*h/(2.*p*epsilon), W)
	# Avoiding possible problems at machine prec. level, e.g. in ex. 9 at [0, 0]
	peclet_array = np.absolute(peclet.vector().array())
	function_of_peclet = np.subtract(coth(peclet_array),np.power(peclet_array,-1))
	f_peclet = Function(W)
	f_peclet.vector()[:] = function_of_peclet
	tau = project((h/(2.*p*conditional(lt(bb,1),1,bb**0.5))*f_peclet), W)
	return tau

# SUPG method to solve convection-diffusion-reaction equation
def solve_supg(V, bcs, epsilon, b, c, f, tau):
  u = TrialFunction(V)
  v = TestFunction(V)
  a = (epsilon*dot(grad(u),grad(v)) + v*dot(b,grad(u)) + c*u*v)*dx +\
      inner(-epsilon*div(grad(u))+dot(b,grad(u))+c*u,tau*dot(b,grad(v)))*dx
  L = f*v*dx + inner(f,tau*dot(b,grad(v)))*dx
  uh = Function(V)
  solve(a == L, uh, bcs)
  return uh

################################################################################

def compute_sold_iso_b_parallel(mesh, V, B, uh, b):
  degree = V.ufl_element().degree()
  family = V.ufl_element()._short_name
  V_grad_norm = FunctionSpace(mesh, "DG", degree-1)
  V_grad_squared = FunctionSpace(mesh, "DG", 2*(degree-1))
  
  # H^1 Seminorm of uh
  grad_uh_squared = project(dot(grad(uh),grad(uh)), V_grad_squared)
  grad_uh_squared_array = grad_uh_squared.vector().array()
  grad_uh_squared_array = np.clip(grad_uh_squared_array, 0.001, 100000.) # For computations
  grad_uh_squared.vector()[:] = grad_uh_squared_array
  b_parallel = project(dot(b,grad(uh))/grad_uh_squared*grad(uh), B)
  return b_parallel

def compute_sold_iso_sigma(mesh, V, B, W, uh, iso_u_0, h, b, b_parallel):
  degree = V.ufl_element().degree()
  family = V.ufl_element()._short_name
  V_grad_norm = FunctionSpace(mesh, "DG", degree-1)
  V_grad_squared = FunctionSpace(mesh, "DG", 2*(degree-1))
  
  # H^1 Seminorm of uh
  grad_uh_squared = project(dot(grad(uh),grad(uh)), V_grad_squared)
  grad_uh_squared_array = grad_uh_squared.vector().array()
  grad_uh_squared_array = np.clip(grad_uh_squared_array, 0.001, 100000.) # For computations
  grad_uh_squared_array = np.sqrt(grad_uh_squared_array)
  grad_uh_squared.vector()[:] = grad_uh_squared_array
  grad_uh = project(grad_uh_squared, V_grad_norm)
  
  # L^1 norm of b_parallel
  b_parallel_norm_squared = project(dot(b_parallel,b_parallel), V_grad_squared)
  b_parallel_norm_squared_array = b_parallel_norm_squared.vector().array()
  b_parallel_norm_squared_array = np.clip(b_parallel_norm_squared_array, 0.001, 100000.) # For computations
  b_parallel_norm_squared_array = np.sqrt(b_parallel_norm_squared_array)
  b_parallel_norm_squared.vector()[:] = b_parallel_norm_squared_array
  b_parallel_norm = project(b_parallel_norm_squared, V_grad_norm)
  
  # L^1 norm of b
  b_norm_squared = project(dot(b,b), V_grad_squared)
  b_norm_squared_array = b_norm_squared.vector().array()
  b_norm_squared_array = np.clip(b_norm_squared_array, 0.001, 100000.) # For computations
  b_norm_squared_array = np.sqrt(b_norm_squared_array)
  b_norm_squared.vector()[:] = b_norm_squared_array
  b_norm = project(b_norm_squared, V_grad_norm)
  
  # Eta
  eta_of_bs = project(eta(b_parallel_norm/b_norm), W)
  eta_of_bs_array = eta_of_bs.vector().array()
  eta_of_bs_array = np.clip(eta_of_bs_array, 0., 100000.) # For computations
  eta_of_bs.vector()[:] = eta_of_bs_array
  
  sigma = project(h*h*eta_of_bs*grad_uh/2./b_parallel_norm/iso_u_0, W)
  sigma_array = sigma.vector().array()
  sigma_array = np.clip(sigma_array, 0., 100000.) # For computations
  sigma.vector()[:] = sigma_array
  return sigma

# SOLD method - isotropic diffusion to solve convection-diffusion-reaction equation
def solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma):
  u = TrialFunction(V)
  v = TestFunction(V)
  a = (epsilon*dot(grad(u),grad(v)) + v*dot(b,grad(u)) + c*u*v)*dx +\
      inner(-epsilon*div(grad(u))+dot(b,grad(u))+c*u,tau*dot(b,grad(v)))*dx +\
      inner(-epsilon*div(grad(u))+dot(b,grad(u))+c*u,sigma*dot(b_parallel,grad(v)))*dx
  L = f*v*dx + inner(f,tau*dot(b,grad(v)))*dx + inner(f,sigma*dot(b_parallel,grad(v)))*dx
  uh = Function(V)
  solve(a == L, uh, bcs)
  return uh

# SOLD method - isotropic diffusion - compute the residue
def residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma):
  v = TestFunction(V)
  a = Function(V)
  a.vector()[:] = assemble((epsilon*dot(grad(uh),grad(v)) + v*dot(b,grad(uh)) + c*uh*v)*dx +\
      inner(-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh,tau*dot(b,grad(v)))*dx +\
      inner(-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh,sigma*dot(b_parallel,grad(v)))*dx -\
      f*v*dx - inner(f,tau*dot(b,grad(v)))*dx - inner(f,sigma*dot(b_parallel,grad(v)))*dx)
  return norm(a)

# Iteration, returns sigma
def iterate_sold_iso(mesh, V, B, W, bcs, iso_u_0, h, epsilon, b, c, f, tau, uh0, tol):
  uh = uh0
  residue = tol + 1.0
  while residue > tol:
    b_parallel = compute_sold_iso_b_parallel(mesh, V, B, uh, b)
    sigma = compute_sold_iso_sigma(mesh, V, B, W, uh, iso_u_0, h, b, b_parallel)
    residue = residue_sold_iso(V, uh, epsilon, b, b_parallel, c, f, tau, sigma)
    uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma)
    print(residue)
  return sigma, b_parallel

################################################################################

# Compute the Tau Parameter From SOLD Method proposed by Codina
def compute_sold_tau_codina(uh, codina_c, W, h, epsilon, b, c, f):
	# H^1 Seminorm of uh
	grad_uh = project(dot(grad(uh),grad(uh)), W)
	
	plot(grad_uh)
	
	grad_uh_array = grad_uh.vector().array()
	grad_uh_array = np.clip(grad_uh_array, 1., 1000.) # For future computations...
	grad_uh_array = np.sqrt(grad_uh_array)
	grad_uh.vector()[:] = grad_uh_array

	# Computation of Res(uh)
	res = project(
		(-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f)**2, W
	)
	res_array = res.vector().array()
	res_array = np.clip(res_array, 0., 100.)
	res_array = np.sqrt(res_array)
	res.vector()[:] = res_array

	# Computation of Q_K(uh)
	lower_bound_q_k = project(
		2*epsilon/codina_c/h, W
	)
	l_b_const = max(lower_bound_q_k.vector())
	q_k = project(
		res/grad_uh, W
	)
	q_k_array = q_k.vector().array()
	q_k_array = np.clip(q_k_array, l_b_const, 100.)
	q_k.vector()[:] = q_k_array

	# Computation of tau
	tau = project(
		(codina_c-2*epsilon/q_k/h)*h*q_k, W
	)
	plot(tau)
	"""
	print min(q_k.vector())
	b_grad_abs = project(
		conditional(
			gt(grad_uh, 0.05), dot(b,grad(uh))/grad_uh, 0.
		), W
	)
	plot(b_grad_abs)

	res = project(
		sqrt((-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f)**2), W
	)
	plot(res)
	b_grad_abs_rectified = project(
		conditional(
			gt(b_grad_abs, epsilon/codina_c/h), b_grad_abs, epsilon/codina_c/h
		), W
	)
	cond_part_second = project(
		codina_c-(2.*epsilon/b_grad_abs_rectified/h), W
	)
	cond_part = project(
		conditional(
			gt(cond_part_second, 0.), cond_part_second, 0.
		), W
	)
	#plot(cond_part)
	
	res = project(
		sqrt(inner((-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f),(-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f))), W
	)
	"""
	return tau

# SOLD method (version 1) to solve convection-diffusion-reaction equation
def solve_sold(V, bcs, epsilon, b, b_perp, c, f, tau, tau2):
  u = TrialFunction(V)
  v = TestFunction(V)
  a = (epsilon*dot(grad(u),grad(v)) + v*dot(b,grad(u)) + c*u*v)*dx +\
      inner(-epsilon*div(grad(u))+dot(b,grad(u))+c*u,tau*dot(b,grad(v)))*dx +\
      inner(dot(b_perp,grad(u)),tau2*dot(b_perp,grad(v)))*dx
  L = f*v*dx + inner(f,tau*dot(b,grad(v)))*dx
  uh = Function(V)
  solve(a == L, uh, bcs)
  return uh

################################################################################

# Error Indicator With Added Crosswind Derivative Control Term
def value_of_ind_cross(V, cut_b_elem_dofs, bcs, epsilon, b, b_perp, c, f, tau):
	fcn_in_ind = lambda u:conditional(gt(u,1), sqrt(u), 2.5*u**2 - 1.5*u**3)
	v = TestFunction(V)
	uh = solve_supg(V, bcs, epsilon, b, c, f, tau)
	# Indicator
	error = assemble(
	  ((-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f)**2
	    + 1. * fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-17)
	  )*cut_b_elem_dofs*dx)
	return error

def value_of_ind_cross_sold(V, cut_b_elem_dofs, bcs, epsilon, b, b_perp, c, f, tau, tau2):
	fcn_in_ind = lambda u:conditional(gt(u,1), sqrt(u), 2.5*u**2 - 1.5*u**3)
	v = TestFunction(V)
	uh = solve_sold(V, bcs, epsilon, b, b_perp, c, f, tau, tau2)
	# Indicator
	error = assemble(
	  ((-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f)**2
	    + 1. * fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-17)
	  )*cut_b_elem_dofs*dx)
	return error

def value_of_ind_cross_sold_iso(V, cut_b_elem_dofs, bcs, epsilon, b, b_perp, b_parallel, c, f, tau, sigma):
	fcn_in_ind = lambda u:conditional(gt(u,1), sqrt(u), 2.5*u**2 - 1.5*u**3)
	v = TestFunction(V)
	uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma)
	# Indicator
	error = assemble(
	  ((-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f)**2
	    + 1. * fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-17)
	  )*cut_b_elem_dofs*dx)
	return error

def der_of_ind_cross(V, W, cut_b_elem_dofs, bcs, bc_V_zero,
		epsilon, b, b_perp, c, f, tau):
	der_of_fcn_in_ind = lambda u:conditional(gt(u,1),0.5/sqrt(u),5.*u-4.5*u**2)
	psi = TrialFunction(V)
	v = TestFunction(V)
	w = TestFunction(W)
	uh = solve_supg(V, bcs, epsilon, b, c, f, tau)
	# Derivatives
	derivatives_assemble = assemble((2*inner(
	  (-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f),
	  (-epsilon*div(grad(v ))+dot(b,grad(v ))+c*v))
	    + 1. * der_of_fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-17)
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

def der_of_ind_cross_sold(V, W, cut_b_elem_dofs, bcs, bc_V_zero,
		epsilon, b, b_perp, c, f, tau, tau2):
	der_of_fcn_in_ind = lambda u:conditional(gt(u,1),0.5/sqrt(u),5.*u-4.5*u**2)
	psi = TrialFunction(V)
	psi2 = TrialFunction(V)
	v = TestFunction(V)
	w = TestFunction(W)
	uh = solve_sold(V, bcs, epsilon, b, b_perp, c, f, tau, tau2)
	# Derivatives
	derivatives_assemble = assemble((2*inner(
	  (-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f),
	  (-epsilon*div(grad(v ))+dot(b,grad(v ))+c*v))
	    + 1. * der_of_fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-17)
	                          * sign(dot(b_perp,grad(uh))) * dot(b_perp,grad(v))
	  )*cut_b_elem_dofs*dx)
	derivatives = Function(V, derivatives_assemble)
	# Adjoint Approach To Compute Derivatives According To tau or tau2
	a = (epsilon*dot(grad(v),grad(psi)) + psi*dot(b,grad(v)) + c*v*psi)*dx +\
	    inner(-epsilon*div(grad(v))+dot(b,grad(v))+c*v,tau*dot(b,grad(psi)))*dx +\
	    inner(dot(b_perp,grad(v)),tau2*dot(b_perp,grad(psi)))*dx
	L = derivatives*v*dx
	psih = Function(V)
	solve(a == L, psih, bc_V_zero)
	# Compute D_Phi_h SUPG tau
	D_Phi_h_ufl_supg = (-inner(-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh,
		w*dot(b,grad(psih))) + inner(f,w*dot(b,grad(psih))))*dx
	D_Phi_h_assemble_supg = assemble(D_Phi_h_ufl_supg)
	D_Phi_h_supg = Function(W, D_Phi_h_assemble_supg)
	# Compute D_Phi_h SOLD tau2
	D_Phi_h_ufl_sold = (-inner(dot(b_perp,grad(uh)),
		w*dot(b_perp,grad(psih))))*dx
	D_Phi_h_assemble_sold = assemble(D_Phi_h_ufl_sold)
	D_Phi_h_sold = Function(W, D_Phi_h_assemble_sold)
	return D_Phi_h_supg, D_Phi_h_sold

def der_of_ind_cross_sold_iso(V, W, cut_b_elem_dofs, bcs, bc_V_zero,
		epsilon, b, b_perp, b_parallel, c, f, tau, sigma):
	der_of_fcn_in_ind = lambda u:conditional(gt(u,1),0.5/sqrt(u),5.*u-4.5*u**2)
	psi = TrialFunction(V)
	psi2 = TrialFunction(V)
	v = TestFunction(V)
	w = TestFunction(W)
	uh = solve_sold_iso(V, bcs, epsilon, b, b_parallel, c, f, tau, sigma)
	# Derivatives
	derivatives_assemble = assemble((2*inner(
	  (-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f),
	  (-epsilon*div(grad(v ))+dot(b,grad(v ))+c*v))
	    + 1. * der_of_fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-17)
	                          * sign(dot(b_perp,grad(uh))) * dot(b_perp,grad(v))
	  )*cut_b_elem_dofs*dx)
	derivatives = Function(V, derivatives_assemble)
	# Adjoint Approach To Compute Derivatives According To tau or sigma
	a = (epsilon*dot(grad(v),grad(psi)) + psi*dot(b,grad(v)) + c*v*psi)*dx +\
	    inner(-epsilon*div(grad(v))+dot(b,grad(v))+c*v,tau*dot(b,grad(psi)))*dx +\
	    inner(-epsilon*div(grad(v))+dot(b,grad(v))+c*v,sigma*dot(b_parallel,grad(psi)))*dx
	L = derivatives*v*dx
	psih = Function(V)
	solve(a == L, psih, bc_V_zero)
	# Compute D_Phi_h SUPG tau
	D_Phi_h_ufl_supg = (-inner(-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh,
		w*dot(b,grad(psih))) + inner(f,w*dot(b,grad(psih))))*dx
	D_Phi_h_assemble_supg = assemble(D_Phi_h_ufl_supg)
	D_Phi_h_supg = Function(W, D_Phi_h_assemble_supg)
	# Compute D_Phi_h SOLD sigma
	D_Phi_h_ufl_sold = (-inner(-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh,
		w*dot(b_parallel,grad(psih))) + inner(f,w*dot(b_parallel,grad(psih))))*dx
	D_Phi_h_assemble_sold = assemble(D_Phi_h_ufl_sold)
	D_Phi_h_sold = Function(W, D_Phi_h_assemble_sold)
	return D_Phi_h_supg, D_Phi_h_sold
