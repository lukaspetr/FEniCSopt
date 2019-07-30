from dolfin import *

import math

def sc_setup(V, sc_example):

  if sc_example == 1:
    # Boundary conditions
    def right(x, on_boundary): return x[0] > (1. - DOLFIN_EPS)
    def left(x, on_boundary): return x[0] < DOLFIN_EPS
    def bottom_center(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 1./3. - DOLFIN_EPS and x[0] < 2./3. + DOLFIN_EPS)
    def bottom_l(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] < 1./3. + DOLFIN_EPS)
    def bottom_r(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 2./3. - DOLFIN_EPS)
    def top(x, on_boundary):
      return x[1] > 1. - DOLFIN_EPS
    g0 = Constant(0.)
    g1 = Constant(1.)
    gl = Expression('x[0]', degree=1)
    gc = Expression('1./3.+x[0]', degree=1)
    gr = Expression('1.-x[0]', degree=1)
    bcl = DirichletBC(V, gl, bottom_l)
    bcc = DirichletBC(V, gc, bottom_center)
    bcr = DirichletBC(V, gr, bottom_r)
    bc3 = DirichletBC(V, g0, top)
    bc4 = DirichletBC(V, g0, right)
    bcs = [bcl, bcc, bcr, bc3, bc4]
    # Data
    epsilon = Constant(1.e-8)
    c = Constant(0.)
    b = Expression(('-x[1]', 'x[0]'), degree=1)
    f = Constant(0.)
    class U_exact_1(UserExpression):
      def eval(self, value, x):
        r = x[0]*x[0]+x[1]*x[1]
        if (r < 1./9.):
          value[0] = sqrt(r)
        elif (r >= 1./9. and r <= 4./9.):
          value[0] = 1./3. + sqrt(r)
        elif (r >= 4./9. and r < 1.):
          value[0] = 1. - sqrt(r)
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_1(degree=1))

  if sc_example == 2:
    # Boundary conditions
    def right(x, on_boundary): return x[0] > (1. - DOLFIN_EPS)
    def left(x, on_boundary): return x[0] < DOLFIN_EPS
    
    def bottom_1(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] < 1./6. + DOLFIN_EPS)
    def bottom_2(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 1./6. - DOLFIN_EPS and x[0] < 2./6. + DOLFIN_EPS)
    def bottom_3(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 2./6. - DOLFIN_EPS and x[0] < 3./6. + DOLFIN_EPS)
    def bottom_4(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 3./6. - DOLFIN_EPS and x[0] < 4./6. + DOLFIN_EPS)
    def bottom_5(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 4./6. - DOLFIN_EPS and x[0] < 5./6. + DOLFIN_EPS)
    def bottom_6(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 5./6. - DOLFIN_EPS)
    def top(x, on_boundary):
      return x[1] > 1. - DOLFIN_EPS
    g0 = Constant(0.)
    g1 = Constant(1.)

    b1 = Expression('6.*x[0]', degree=1)
    b2 = Expression('2.-6.*x[0]', degree=1)
    b3 = Expression('-2.+6.*x[0]', degree=1)
    b4 = Expression('4.-6.*x[0]', degree=1)
    b5 = Expression('-4.+6.*x[0]', degree=1)
    b6 = Expression('6.-6.*x[0]', degree=1)

    bcb1 = DirichletBC(V, b1, bottom_1)
    bcb2 = DirichletBC(V, b2, bottom_2)
    bcb3 = DirichletBC(V, b3, bottom_3)
    bcb4 = DirichletBC(V, b4, bottom_4)
    bcb5 = DirichletBC(V, b5, bottom_5)
    bcb6 = DirichletBC(V, b6, bottom_6)

    bc3 = DirichletBC(V, g0, top)
    bc4 = DirichletBC(V, g0, right)

    bcs = [bcb1, bcb2, bcb3, bcb4, bcb5, bcb6, bc3, bc4]

    # Data
    epsilon = Constant(1.e-8)
    c = Constant(0.)
    b = Expression(('-x[1]', 'x[0]'), degree=1)
    f = Constant(0.)
    class U_exact_1(UserExpression):
      def eval(self, value, x):
        r = x[0]*x[0]+x[1]*x[1]
        if (r < 1./36.):
          value[0] = 6.*sqrt(r)
        elif (r >= 1./36. and r < 4./36.):
          value[0] = 2. - 6.*sqrt(r)
        elif (r >= 4./36. and r < 9./36.):
          value[0] = -2. + 6.*sqrt(r)
        elif (r >= 9./36. and r < 16./36.):
          value[0] = 4. - 6.*sqrt(r)
        elif (r >= 16./36. and r < 25./36.):
          value[0] = -4. + 6.*sqrt(r)
        elif (r >= 25./36. and r < 1.):
          value[0] = 6. - 6.*sqrt(r)
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_1(degree=1))

  if sc_example == 3:
    # Boundary conditions
    def right(x, on_boundary): return x[0] > (1. - DOLFIN_EPS)
    def left(x, on_boundary): return x[0] < DOLFIN_EPS
    def top(x, on_boundary): return x[1] > (1. - DOLFIN_EPS)

    def bottom_1(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] < 1./6. + DOLFIN_EPS)
    def bottom_2(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 1./6. - DOLFIN_EPS and x[0] < 2./3. + DOLFIN_EPS)
    def bottom_3(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 2./3. - DOLFIN_EPS and x[0] < 5./6. + DOLFIN_EPS)
    def bottom_4(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 5./6. - DOLFIN_EPS)

    g0 = Constant(0.)
    g1 = Constant(1.)

    b2 = Expression('sqrt(-7./36.-x[0]*x[0]+4./3.*x[0])', degree=2)
    b3 = Expression('2.5-3.*x[0]', degree=1)

    bcb1 = DirichletBC(V, g0, bottom_1)
    bcb2 = DirichletBC(V, b2, bottom_2)
    bcb3 = DirichletBC(V, g0, bottom_3)
    bcb4 = DirichletBC(V, g0, bottom_4)

    bc3 = DirichletBC(V, g0, top)
    bc4 = DirichletBC(V, g0, right)

    bcs = [bcb1, bcb2, bcb3, bcb4, bc3, bc4]

    # Data
    epsilon = Constant(1.e-8)
    c = Constant(0.)
    b = Expression(('-x[1]', 'x[0]'), degree=1)
    f = Constant(0.)
    class U_exact_1(UserExpression):
      def eval(self, value, x):
        r = x[0]*x[0]+x[1]*x[1]
        if (r < 1./36.):
          value[0] = 0.
        elif (r >= 1./36. and r < 4./9.):
          value[0] = sqrt(-7./36.-r+4./3.*sqrt(r))
        elif (r >= 16./36. and r < 25./36.):
          value[0] = 0.
        elif (r >= 25./36. and r < 1.):
          value[0] = 0.
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_1(degree=1))

  if sc_example == 4:
    # Boundary conditions
    def right(x, on_boundary): return x[0] > (1. - DOLFIN_EPS)
    def left(x, on_boundary): return x[0] < DOLFIN_EPS
    def top(x, on_boundary): return x[1] > (1. - DOLFIN_EPS)

    def bottom_1(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] < 1./6. + DOLFIN_EPS)
    def bottom_2(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 1./6. - DOLFIN_EPS and x[0] < 2./3. + DOLFIN_EPS)
    def bottom_3(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 2./3. - DOLFIN_EPS and x[0] < 5./6. + DOLFIN_EPS)
    def bottom_4(x, on_boundary):
      return x[1] < DOLFIN_EPS and (x[0] > 5./6. - DOLFIN_EPS)

    g0 = Constant(0.)
    g1 = Constant(1.)

    b2 = Expression('sqrt(-7./36.-x[0]*x[0]+4./3.*x[0])', degree=2)
    b3 = Expression('2.5-3.*x[0]', degree=1)

    bcb1 = DirichletBC(V, g0, bottom_1)
    bcb2 = DirichletBC(V, b2, bottom_2)
    bcb3 = DirichletBC(V, g0, bottom_3)
    bcb4 = DirichletBC(V, g0, bottom_4)

    bc3 = DirichletBC(V, g0, top)
    bc4 = DirichletBC(V, g0, right)

    bcs = [bcb1, bcb2, bcb3, bcb4, bc3, bc4]

    # Data
    epsilon = Constant(1.e-8)
    c = Constant(0.)
    b = Expression(('0', '1'), degree=1)
    f = Constant(0.)
    class U_exact_1(UserExpression):
      def eval(self, value, x):
        if x[1] < (1-DOLFIN_EPS):
          r = x[0]*x[0]
          if (r < 1./36.):
            value[0] = 0.
          elif (r >= 1./36. and r < 4./9.):
            value[0] = sqrt(-7./36.-r+4./3.*sqrt(r))
          elif (r >= 16./36. and r < 25./36.):
            value[0] = 0.
          elif (r >= 25./36. and r < 1.):
            value[0] = 0.
          else:
            value[0] = 0.
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_1(degree=1))

  if sc_example == 8:
    # Boundary conditions
    def zero(x, on_boundary):
      return on_boundary and (x[0] > (1. - DOLFIN_EPS) or  x[1] < 0.7)
    def one (x, on_boundary):
      return on_boundary and (x[0] < (1. - DOLFIN_EPS) and x[1] >= 0.7)
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
    class U_exact_8(UserExpression):
      def eval(self, value, x):
        if (x[0] > 0.0001 and x[0] < 0.9999 and x[1] > 0.0001 and x[1] < 0.9999):
          if (x[1] > (0.7 + x[0]*math.sin(theta)/math.cos(theta))):
            value[0] = 1.
          else:
            value[0] = 0.
        elif ((x[0] < 1.e-10 and x[1] > 0.7) or (x[1] > 0.999999 and x[0] < 0.999999)):
          value[0] = 1.
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_8(degree=1))

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
    b = Expression(('-x[1]', 'x[0]'), degree=1)
    f = Constant(0.)
    class U_exact_9(UserExpression):
      def eval(self, value, x):
        r = x[0]*x[0]+x[1]*x[1]
        if (r > 1./9. and r < 4./9.):
          value[0] = 1.
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_9(degree=1))

  elif sc_example == 20:
    # Boundary conditions
    def whole_boundary(x, on_boundary):
      return on_boundary
    bcs = DirichletBC(V, 0., whole_boundary)
    # Data
    epsilon = Constant(1.e-8)
    c = Constant(0.)
    b = Constant((1., 0.))
    class RhsFunction(UserExpression):
      def eval(self, value, x):
        if (abs(x[0]-0.5) > 0.25 or abs(x[1]-0.5) > 0.25):
          value[0] = 0.
        else:
          value[0] = -32.*(x[0]-0.5)
    f = Function(V)
    f.interpolate(RhsFunction(degree=1))
    class U_exact_20(UserExpression):
      def eval(self, value, x):
        if (abs(x[0]-0.5) > 0.25 or abs(x[1]-0.5) > 0.25):
          value[0] = 0.
        else:
          value[0] = -16.*(x[0]-0.25)*(x[0]-0.75)
    u_exact = Function(V)
    u_exact.interpolate(U_exact_20(degree=1))

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
    class U_exact_55(UserExpression):
      def eval(self, value, x):
        if (x[0] > 1.e-6 and x[0] < 0.999999 and x[1] > 1.e-6 and x[1] < 0.999999):
          value[0] = x[0]
        else:
          value[0] = 0.
    u_exact = Function(V)
    u_exact.interpolate(U_exact_55(degree=1))

  return bcs, epsilon, c, b, f, u_exact
