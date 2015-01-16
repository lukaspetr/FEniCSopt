from dolfin import *

import math

def sc_setup(V, sc_example):

  if sc_example == 8:
    # Boundary conditions
    def zero(x, on_boundary):
      return on_boundary and (x[0] > (1. - DOLFIN_EPS) or  x[1] < 0.7)
    def one (x, on_boundary):
      return on_boundary and (x[0] < (1. - DOLFIN_EPS) and x[1] > 0.7)
    g0 = Constant(0.)
    g1 = Constant(1.)
    bc0 = DirichletBC(V, g0, zero)
    bc1 = DirichletBC(V, g1, one)
    bcs = [bc0, bc1]
    # Data
    epsilon = Constant(1.e-8)
    c = Constant(0.)
    theta = -math.pi/3.
    b = Constant((math.cos(theta), math.sin(theta)))
    f = Constant(0.)
    class U_exact_8(Expression):
      def eval(self, value, x):
        if (x[0] > 0.0001 and x[0] < 0.9999 and x[1] > 0.0001 and x[1] < 0.9999):
          if (x[1] > (0.7 + x[0]*math.cos(theta)/math.sin(theta))):
            value[0] = 1.
          else:
            value[0] = 0.
        elif ((x[0] < 1.e-10 and x[1] > 0.7) or (x[1] > 0.999999 and x[0] < 0.999999)):
          value[0] = 1.
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_8())

  if sc_example == 9:
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
    b = Expression(('-x[1]', 'x[0]'))
    f = Constant(0.)
    class U_exact_9(Expression):
      def eval(self, value, x):
        r = x[0]*x[0]+x[1]*x[1]
        if (r > 1./9. and r < 4./9.):
          value[0] = 1.
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_9())

  elif sc_example == 20:
    # Boundary conditions
    def whole_boundary(x, on_boundary):
      return on_boundary
    bcs = DirichletBC(V, 0., whole_boundary)
    # Data
    epsilon = Constant(1.e-8)
    c = Constant(0.)
    b = Constant((1., 0.))
    class RhsFunction(Expression):
      def eval(self, value, x):
        if (abs(x[0]-0.5) > 0.25 or abs(x[1]-0.5) > 0.25):
          value[0] = 0.
        else:
          value[0] = -32.*(x[0]-0.5)
    f = Function(V)
    f.interpolate(RhsFunction())
    class U_exact_20(Expression):
      def eval(self, value, x):
        if (abs(x[0]-0.5) > 0.25 or abs(x[1]-0.5) > 0.25):
          value[0] = 0.
        else:
          value[0] = -16.*(x[0]-0.25)*(x[0]-0.75)
    u_exact = Function(V)
    u_exact.interpolate(U_exact_20())

  elif sc_example == 55:
    # Boundary conditions
    def whole_boundary(x, on_boundary):
      return on_boundary
    bcs = DirichletBC(V, 0., whole_boundary)
    # Data
    epsilon = Constant(1.e-8)
    c = Constant(0.)
    b = Constant((1., 0.))
    f = Constant(1.)
    class U_exact_55(Expression):
      def eval(self, value, x):
        if (x[0] > 1.e-6 and x[0] < 0.999999 and x[1] > 1.e-6 and x[1] < 0.999999):
          value[0] = x[0]
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_55())

  return bcs, epsilon, c, b, f, u_exact
