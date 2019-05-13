from dolfin import *
import numpy as np
coth = lambda x: 1./np.tanh(x)

# Mesh and Function Spaces
mesh = UnitSquareMesh(10,10)
h = CellDiameter(mesh)
V = FunctionSpace(mesh, "CG", 4)
v = TestFunction(V)
W = FunctionSpace(mesh, "DG", 6)
w = TestFunction(W)

# Boundary conditions
def right(x, on_boundary): return x[0] > (1. - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def bottom_center(x, on_boundary):
  return x[1] < DOLFIN_EPS and (x[0] > 1./3. - DOLFIN_EPS and x[0] < 2./3. + DOLFIN_EPS)
def bottom_lr(x, on_boundary):
  return x[1] < DOLFIN_EPS and (x[0] < 1./3. + DOLFIN_EPS or x[0] > 2./3. - DOLFIN_EPS)
def top(x, on_boundary):
  return x[1] > 1. - DOLFIN_EPS
g0 = Constant(0.)
g1 = Constant(1.)
bc0 = DirichletBC(V, g1, bottom_center)
bc1 = DirichletBC(V, g0, bottom_lr)
bc3 = DirichletBC(V, g0, top)
bc4 = DirichletBC(V, g0, right)
bcs = [bc0, bc1, bc4, bc3]

# Data
epsilon = Constant(1.e-8)
c = Constant(0.)
b = Expression(('-x[1]', 'x[0]'), degree=1)
f = Constant(0.)

# Compute the Tau Parameter From SUPG Method
def compute_tau(W, h, p, epsilon, b):
	bb = project(dot(b,b), W)
	# Tau Definition (Peclet Number And The Function Of The Pecl. Number Are Used)
	peclet = project(bb**0.5*h/(2.*p*epsilon), W)
	# Avoiding possible problems at machine prec. level, e.g. in ex. 9 at [0, 0]
	peclet_array = np.absolute(peclet.vector().get_local())
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

# Main Part
tau = compute_tau(W, h, 1., epsilon, b)
uh = solve_supg(V, bcs, epsilon, b, c, f, tau)
plot(uh)

grad_uh_V_assemble = assemble(dot(grad(uh),grad(uh))*v*dx)
grad_uh_V = Function(V, grad_uh_V_assemble)
plot(grad_uh_V)

grad_uh = project(dot(grad(uh),grad(uh)), W)
plot(grad_uh)
