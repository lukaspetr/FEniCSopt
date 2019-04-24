from __future__ import division
from dolfin import *
import numpy
import pprint

# Gnuplot related functions

################################################################################

# discontinuous piecewise linear output
def gnuplot_dg1(file, mesh, fun):
	file = open(file, 'w+')
	i = 0
	for myCell in cells(mesh):
		i += 1
		vs = []
		for v in vertices(myCell):
		    vs.append(v.midpoint())
		
		print('%e  %e  %e' % (vs[0].x(),vs[0].y(),fun(vs[0].x(),vs[0].y())), file=file)
		print('%e  %e  %e' % (vs[1].x(),vs[1].y(),fun(vs[1].x(),vs[1].y())), file=file)
		print('%e  %e  %e' % (vs[2].x(),vs[2].y(),fun(vs[2].x(),vs[2].y())), file=file)
		print('%e  %e  %e' % (vs[0].x(),vs[0].y(),fun(vs[0].x(),vs[0].y())), file=file)
		if (i == 1):
			print('%e  %e  %e' % (vs[0].x(),vs[0].y(),fun(vs[0].x(),vs[0].y())), file=file)
		print('', file=file)

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
			print('%e  %e  %e' % (x,y,f), file=file)
		print('', file=file)

################################################################################

# graph output
def gnuplot_graph(file, data):
	file = open(file, 'w+')
	for point in data:
		pprint.pprint(point)
		print('%e  %e' % (point['position'], point['phi']), file=file)
	print('', file=file)

