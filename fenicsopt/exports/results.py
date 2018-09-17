from __future__ import print_function
import os
from dolfin import *
from gnuplot import *
import numpy as np
from scipy.optimize import curve_fit



# Make All Necessary Results
def make_results(SC_EXAMPLE, NUM_CELL, V, W, uh, u_exact, tau, res, results):

	V_DEGREE = V.ufl_element().degree()
	W_DEGREE = W.ufl_element().degree()
	V_FAMILY = V.ufl_element()._short_name
	W_FAMILY = W.ufl_element()._short_name

	hash = V_FAMILY + str(V_DEGREE) + W_FAMILY + str(W_DEGREE)
	folder = str(SC_EXAMPLE) + '/' + hash + '/'
	if not os.path.exists(folder):
		os.makedirs(folder)
	gnufile = open(folder + hash + '.cmd', 'w')

	# Plot Solution And Exact Solution With Gnuplot
	GNUPLOT_NUM_CELL = V_DEGREE * (NUM_CELL-1) + 1
	gnuplot_mesh = UnitSquareMesh(GNUPLOT_NUM_CELL,GNUPLOT_NUM_CELL)
	gnuplot_V = FunctionSpace(gnuplot_mesh, "CG", 1)
	gnuplot_uh = project(uh, gnuplot_V)
	gnuplot_nodally_exact = project(u_exact, gnuplot_V)
	gnuplot_dg1(folder + hash + '.gnu', gnuplot_mesh, gnuplot_uh)
	gnuplot_dg1(folder + hash + '.exact.gnu', gnuplot_mesh, gnuplot_nodally_exact)
	template = """
		unset logscale x
		unset logscale y
		set xrange [0.0:1.0]
		set yrange [0.0:1.0]
		set zrange [-0.2:1.2]
		set ticslevel 0.0
		set view 63,13
		set size square
		set terminal postscript eps enhanced 18
		set format xy "%g"
		set output "{folder}{hash}.eps"
		sp "{folder}{hash}.gnu" w l title ""
		set output "{folder}{hash}.exact.eps"
		sp "{folder}{hash}.exact.gnu" w l title ""
	"""
	gnufile.write(template.format(folder = folder, hash = hash))

	# Plot Tau On Equidistant Mesh With Gnuplot
	gnuplot_square_equidistant(folder + hash + '.tau.map', 600, tau)
	template = """
		unset logscale x
		unset logscale y
		set pm3d map
		set xrange [0.0:1.0]
		set yrange [0.0:1.0]
		set cbrange [0.0:0.1]
		set size square
		set terminal png transparent interlace truecolor font "Helvetica,36" enhanced size 1600,1360
		set output "{folder}{hash}.tau.png"
		sp "{folder}{hash}.tau.map" title ""
	"""
	gnufile.write(template.format(folder = folder, hash = hash))

	# Print Error Indicator Of Result to a File
	res_file = folder + hash + '.res'
	res_file = open(res_file, 'w+')
	res_file.write(str(res))

	# Print process of minimization
	min_file = folder + hash + '.min'
	min_file = open(min_file, 'w+')
	for result in results:
		min_file.write('{:e}  {:e}\n'.format(result[0], result[1]))
	template = """
		unset yrange
		set yrange [*:*] noreverse
		set logscale x
		set logscale y
		set xrange [1:100]
		set size square
		set terminal postscript eps enhanced 22
		set format y "10^{{%L}}"
		set output "{folder}{hash}.min.eps"
		p "{folder}{hash}.min" w l title "{hash}"
	"""
	gnufile.write(template.format(folder = folder, hash = hash))

	# Call Gnuplot And Other Programs
	popen1 = os.popen('gnuplot ' + folder + hash + '.cmd\n convert '
		+ folder + hash + '.tau.png -crop 1280x1090+200+150 '
		+ folder + hash + '.tau.png\n', 'r')
	# Currently, the following is not used/not working in some cases, non-critical
	popen5 = os.popen('epstool --copy --bbox ' + folder + hash + '.exact.eps '
		+ folder + hash + '.exact.cropped.eps\n', 'r')
	popen6 = os.popen('epstool --copy --bbox ' + folder + hash + '.eps '
		+ folder + hash + '.cropped.eps\n', 'r')



# Make Results For SOLD Parameter
def make_results_sold_par(SC_EXAMPLE, NUM_CELL, V, W, tau):

	V_DEGREE = V.ufl_element().degree()
	W_DEGREE = W.ufl_element().degree()
	V_FAMILY = V.ufl_element()._short_name
	W_FAMILY = W.ufl_element()._short_name

	hash = V_FAMILY + str(V_DEGREE) + W_FAMILY + str(W_DEGREE)
	folder = str(SC_EXAMPLE) + '/' + hash + '/'
	if not os.path.exists(folder):
		os.makedirs(folder)
	gnufile = open(folder + hash + '.cmd', 'w')

	# Plot Tau On Equidistant Mesh With Gnuplot
	gnuplot_square_equidistant(folder + hash + '.tau2.map', 600, tau)
	template = """
		unset logscale x
		unset logscale y
		set pm3d map
		set xtics textcolor rgb "black"
		set ytics textcolor rgb "black"
		set xrange [0.0:1.0]
		set yrange [0.0:1.0]
		set cbrange [0.0:0.1]
		set size square
		set terminal png transparent interlace truecolor font "Helvetica,36" enhanced size 1600,1360
		set output "{folder}{hash}.tau2.png"
		sp "{folder}{hash}.tau2.map" title ""
	"""
	gnufile.write(template.format(folder = folder, hash = hash))

	# Call Gnuplot And Other Programs
	popen1 = os.popen('gnuplot ' + folder + hash + '.cmd\n convert '
		+ folder + hash + '.tau2.png -crop 1280x1090+200+150 '
		+ folder + hash + '.tau2.png\n', 'r')



# Make h Results
def make_results_h(SC_EXAMPLE, NUM_CELL, V, W, uh, u_exact, yh, res_phi):

	V_DEGREE = V.ufl_element().degree()
	W_DEGREE = W.ufl_element().degree()
	V_FAMILY = V.ufl_element()._short_name
	W_FAMILY = W.ufl_element()._short_name

	hash = V_FAMILY + str(V_DEGREE) + W_FAMILY + str(W_DEGREE) + '_' + str(NUM_CELL)
	folder = str(SC_EXAMPLE) + '/' + hash + '/'
	if not os.path.exists(folder):
		os.makedirs(folder)
	gnufile = open(folder + hash + 'h.cmd', 'w')

	# Plot Solution With Gnuplot
	GNUPLOT_NUM_CELL = V_DEGREE * (NUM_CELL-1) + 1
	gnuplot_mesh = UnitSquareMesh(GNUPLOT_NUM_CELL,GNUPLOT_NUM_CELL)
	gnuplot_V = FunctionSpace(gnuplot_mesh, "CG", 1)
	gnuplot_uh = project(uh, gnuplot_V)
	gnuplot_dg1(folder + hash + '.gnu', gnuplot_mesh, gnuplot_uh)
	template = """
		unset logscale x
		unset logscale y
		set xrange [0.0:1.0]
		set yrange [0.0:1.0]
		set zrange [-0.2:1.2]
		set ticslevel 0.0
		set view 63,13
		set size square
		set terminal postscript eps enhanced 18
		set format xy "%g"
		set output "{folder}{hash}.eps"
		sp "{folder}{hash}.gnu" w l title ""
	"""
	gnufile.write(template.format(folder = folder, hash = hash))

	# Call Gnuplot
	popen1 = os.popen('gnuplot ' + folder + hash + 'h.cmd', 'r')



# Make Global Results
def make_global_results(SC_EXAMPLE, global_results):

	folder = str(SC_EXAMPLE) + '/'
	filename = folder + 'dofs.gnu'
	file = open(filename, 'w+')
	filename_30 = folder + 'dofs_30.gnu'
	file_30 = open(filename_30, 'w+')
	
	# Sorting
	sorted_results = sorted(global_results, key=lambda k: (k['V_dofs'], k['W_dofs']))
	last_V_dofs = 0
	for result in sorted_results:
		if last_V_dofs != result['V_dofs']:
			file.write('\n')
			file_30.write('\n')
			last_V_dofs = result['V_dofs']
		file.write('{:e}  {:e}  {:e}\n'.format(result['V_dofs'], result['W_dofs'], result['phi']))
		file_30.write('{:e}  {:e}  {:e}\n'.format(result['V_dofs'], result['W_dofs'], result['phi_30']))
	file.write('\n')
	file_30.write('\n')
	sorted_results = sorted(global_results, key=lambda k: (k['W_dofs'], k['V_dofs']))
	last_W_dofs = 0
	for result in sorted_results:
		if last_W_dofs != result['W_dofs']:
			file.write('\n')
			file_30.write('\n')
			last_W_dofs = result['W_dofs']
		file.write('{:e}  {:e}  {:e}\n'.format(result['V_dofs'], result['W_dofs'], result['phi']))
		file_30.write('{:e}  {:e}  {:e}\n'.format(result['V_dofs'], result['W_dofs'], result['phi_30']))

	template = """
		set size square
		set xtics 0,5000,40000 offset -2, -0.1
		set ytics 0,10000,40000 offset 0.5, -0.5
		set terminal postscript eps enhanced 22
		set output "{folder}dofs.eps"
		sp "{filename}" w l title ""
		set output "{folder}dofs_30.eps"
		sp "{filename_30}" w l title ""
	"""
	gnufile = open(folder + 'gnu.cmd', 'w')
	gnufile.write(template.format(folder = folder, filename = filename,
	                                               filename_30 = filename_30))

	# Call Gnuplot
	popen2 = os.popen('gnuplot ' + folder + 'gnu.cmd', 'r')



# Make Global Results h
def make_global_results_h(SC_EXAMPLE, global_results):

	folder = str(SC_EXAMPLE) + '/'
	filename = folder + 'h.gnu'
	file = open(filename, 'w+')
	
	# Sorting
	sorted_results = sorted(global_results, key=lambda k: (k['h']))
	h_values = np.empty([1])
	error_values = np.empty([1])
	for result in sorted_results:
		file.write('{:e}  {:e} \n'.format(result['h'], result['error_l2']))
		h_values = np.append(h_values, result['h'])
		error_values = np.append(error_values, result['error_l2'])
	file.write('\n')

	def func(x, a, b):
		return a*x**b
	popt, pcov = curve_fit(func, h_values, error_values)
	template = """
		The resulting function of h is:
		
		f(x) = {a}*h**{b}
	"""
	resultsfile = open(folder + 'fit.res', 'w')
	resultsfile.write(template.format(a = popt[0], b = popt[1]))

	template = """
		set logscale x
		set logscale y
		f(x) = a*x**b
		a=2
		b=3
		fit f(x) "{filename}" us 1:2:($2*.01) via a,b
		set size square
		set terminal postscript eps enhanced 22
		set output "{folder}h.eps"
		p f(x) title "fit", "{filename}" title "data"
	"""
	gnufile = open(folder + 'gnu.h.cmd', 'w')
	gnufile.write(template.format(folder = folder, filename = filename, example = SC_EXAMPLE))

	# Call Gnuplot
	popen2 = os.popen('gnuplot ' + folder + 'gnu.h.cmd\n', 'r')
	popen3 = os.popen('epstool --copy --bbox ' + folder + 'h.eps '
		+ folder + 'h.cropped.eps\n')

# Make the line of phi
def make_line_results(SC_EXAMPLE, V, W, results):

	V_DEGREE = V.ufl_element().degree()
	W_DEGREE = W.ufl_element().degree()
	V_FAMILY = V.ufl_element()._short_name
	W_FAMILY = W.ufl_element()._short_name

	hash = V_FAMILY + str(V_DEGREE) + W_FAMILY + str(W_DEGREE)
	folder = str(SC_EXAMPLE) + '/' + hash + '/'
	if not os.path.exists(folder):
		os.makedirs(folder)

	filename = folder + 'line_results.gnu'
	gnuplot_graph(filename, results)
	file = open(folder + 'gnu_line.cmd', 'w+')
	template = """
		set terminal postscript eps enhanced 22
		set output "{folder}line.eps"
		p "{filename}" w l title "data"
	"""
	gnufile = open(folder + 'gnu_line.cmd', 'w')
	gnufile.write(template.format(folder = folder, filename = filename, example = SC_EXAMPLE))

	# Call Gnuplot
	popen2 = os.popen('gnuplot ' + folder + 'gnu_line.cmd\n', 'r')
	popen3 = os.popen('epstool --copy --bbox ' + folder + 'line.eps '
		+ folder + 'line.cropped.eps\n')

