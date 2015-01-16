from dolfin import *
import numpy as np
coth = lambda x: 1./np.tanh(x)

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
	# Tau Definition (Peclet Number And The Function Of The Pecl. Number Are Used)
	peclet = project(bb**0.5*h/(2.*p*epsilon), W)
	# Avoiding possible problems at machine prec. level, e.g. in ex. 9 at [0, 0]
	peclet_array = np.absolute(peclet.vector().array())
	function_of_peclet = np.subtract(coth(peclet_array),np.power(peclet_array,-1))
	f_peclet = Function(W)
	f_peclet.vector()[:] = function_of_peclet
	tau = project((h/(2.*p*conditional(lt(bb,1),1,bb**0.5))*f_peclet), W)
	return tau

################################################################################

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

# Error Indicator With Added Crosswind Derivative Control Term
def value_of_ind_cross(V, cut_b_elem_dofs, bcs, epsilon, b, b_perp, c, f, tau):
	fcn_in_ind = lambda u:conditional(gt(u,1), sqrt(u), 2.5*u**2 - 1.5*u**3)
	v = TestFunction(V)
	uh = solve_supg(V, bcs, epsilon, b, c, f, tau)
	# Indicator
	error = assemble(
	  ((-epsilon*div(grad(uh))+dot(b,grad(uh))+c*uh-f)**2
	    + 0.5 * fcn_in_ind(abs(dot(b_perp,grad(uh)))+1e-15)
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
