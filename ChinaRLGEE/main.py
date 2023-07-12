import os
import sys
import time

script_path = os.path.abspath(__file__) # current script path
package_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(package_path)

from DecimalIndividual import DecimalIntegerIndividual
from Population import Population
from Selection import RouletteWheelSelection
from Crossover import DecimalCrossover
from Mutation import DecimalMutation
from GAProcess import GA

# from functions import *
import numpy as np


class Booth:
	def __init__(self):
		self.objective = lambda x: (x[0]+2*x[1]-7)**2 + (2*x[2]+x[3]-5)**2
		self.ranges = [(-10,10)] * 4
		self.solution = [(1.0,3.0,1,2)]

	@property
	def value(self):
		return self.objective(np.array(self.solution[0]))


def main():

	s = time.time()

	# objective function
	# f = Booth()
	# f = FUN()

	# GA process
	I = DecimalIntegerIndividual([(0,5)] * 28)
	P = Population(I, 50)
	R = RouletteWheelSelection()
	C = DecimalCrossover([0.2, 0.6], 0.55)
	M = DecimalMutation(0.05)
	g = GA(P, R, C, M)

	# solve
	res = g.run(200)

	with open('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_reg_input2/es','a+') as GA_result:
		print('GA solution input: {0}'.format(res.solution), file=GA_result)
		print('GA solution output: {0}'.format(res.evaluation), file=GA_result)

	print('spent time: {0}\n'.format(time.time()-s))

if __name__ == '__main__':

	main()
	# FUNS = [Bukin, Eggholder, Rosenbrock]
	# for fun in FUNS:
	# 	tst(fun)
