from __future__ import division
from dolfin import *
import numpy

# Gnuplot related functions

################################################################################

# discontinuous piecewise linear output
def gnuplot_dg1(file, mesh, fun):
	file = open(file, 'w+')
	i = 0
	for myCell in cells(mesh):
		i += 1
		myVertices = vertices(myCell)
		it = iter(myVertices)
		myVertex0 = it.next()
		myVertex1 = it.next()
		myVertex2 = it.next()
		print >>file, '%e  %e  %e' % (myVertex0.x(0),myVertex0.x(1),\
				                      fun(myVertex0.x(0),myVertex0.x(1)))
		print >>file, '%e  %e  %e' % (myVertex1.x(0),myVertex1.x(1),\
				                      fun(myVertex1.x(0),myVertex1.x(1)))
		print >>file, '%e  %e  %e' % (myVertex2.x(0),myVertex2.x(1),\
				                      fun(myVertex2.x(0),myVertex2.x(1)))
		print >>file, '%e  %e  %e' % (myVertex0.x(0),myVertex0.x(1),\
				                      fun(myVertex0.x(0),myVertex0.x(1)))
		if (i == 1):
			print >>file, '%e  %e  %e' % (myVertex0.x(0),myVertex0.x(1),\
				                        fun(myVertex0.x(0),myVertex0.x(1)))
		print >>file, ''

################################################################################

# for printing with e.g. pm3d map
def gnuplot_square_equidistant(file, N, fun):
	file = open(file, 'w+')
	for i in range(0, N+1):
		for j in range(0, N+1):
			x = i/N
			y = j/N
			p = Point(x,y)
			f = fun(p)
			print >>file, '%e  %e  %e' % (x,y,f)
		print >>file, ''
