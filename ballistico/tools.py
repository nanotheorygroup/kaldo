import subprocess
import os
import numpy as np
import pandas as pd

def run_script(cmd, folder=None):
	# print 'Executing: ' + cmd
	if folder:
		p = subprocess.Popen (cmd, cwd=folder, shell=True, stdout=subprocess.PIPE)
	else:
		p = subprocess.Popen (cmd, shell=True, stdout=subprocess.PIPE)
	(output, err) = p.communicate ()
	p.wait ()
	return output.decode('ascii')
	
def is_folder_present(file_path):
	directory = os.path.dirname (file_path)
	is_present = True
	if not os.path.exists (directory):
		is_present = False
		os.makedirs (directory)
	return is_present

def get_current_folder():
	string = os.getcwd ().replace (' ', '\ ')
	return string
	
def matrix_to_string(matrix):
	string = ''
	if len (matrix.shape) == 1:
		for i in range (matrix.shape[0]):
			string += '%.7f' % matrix[i] + ' '
		string += '\n'
	else:
		for i in range (matrix.shape[0]):
			for j in range (matrix.shape[1]):
				string += '%.7f' % matrix[i, j] + ' '
			string += '\n'
	return string

def function_of_operator(func, operator):

	(eigenvalues, transformation) =  np.linalg.eig(operator)
	func_of_eigenvalues = func(eigenvalues)
	
	operator_to_return = transformation.dot(np.diag(func_of_eigenvalues)).dot(np.linalg.inv(transformation))
	return operator_to_return
	
def read_file(filename):
	file = pd.read_csv (filename, header=None, delim_whitespace=True, )
	read_array = file.values
	read_array = read_array.reshape (read_array.size)
	# check for nan
	# read_array = read_array[~np.isnan (read_array)]
	# if read_array.size == 1:
	# 	read_array = int (read_array)
	return read_array
