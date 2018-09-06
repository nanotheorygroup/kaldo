from data.tersoff import tersoff
import numpy as np

class Potential (object):
	def __init__(self, configuration):
		self.configuration = configuration
		
	def create_potential_string(self, type='tersoff'):
		if type=='tersoff':
			string = self.create_tersoff_string()
		elif type=='neural':
			string = self.create_neural_string()
		return string
	
	def create_neural_string(self):
		unique_elements = np.unique (self.system.configuration.get_chemical_symbols ())
		n_uniques = unique_elements.shape[0]
		string = 'NEURAL ' + str (n_uniques) + '\n'
		for element in unique_elements:
			string +=  element + '\n'
		return string
	
	def create_tersoff_string(self):

		unique_elements = np.unique (self.configuration.get_chemical_symbols ())
		n_uniques = len(unique_elements)
		string = 'TERSOFF ' + str(n_uniques) + '\n'
		for potential in tersoff['potentials']:
			if potential['name'] in unique_elements:
				string += potential['name'] + '    '
				# string += potential['params']
				string += 'ters  '
				string += str (potential['pa']) + '  '
				string += str (potential['pl1']) + '  '
				string += str (potential['pb']) + '  '
				string += str (potential['pl2']) + '  '
				string += str (potential['pr']) + '\n'
				string += str (potential['ps']) + '  '
				string += str (potential['pbt']) + '  '
				string += str (potential['pn']) + '  '
				string += str (potential['pc']) + '  '
				string += str (potential['pd']) + '  '
				string += str (potential['ph']) + '\n'
				
		for element in unique_elements:
			string += element + '  ' + element + '  1.0    1.0\n'
		
		for cross_terms in tersoff['cross_terms']:
			if cross_terms['name_1'] in unique_elements and cross_terms['name_2'] in unique_elements:
				string += cross_terms['name_1'] + '  ' + cross_terms['name_2'] + '  ' + cross_terms['params'] + '\n'
		return string

		