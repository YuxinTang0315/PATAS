#----------------------------------------------------------
# GA Operator: mutation
#----------------------------------------------------------
import numpy as np
from Operators import Mutation
from DecimalIndividual import DecimalFloatIndividual, DecimalIntegerIndividual
from SequenceIndividual import UniqueSeqIndividual, UniqueLoopIndividual, ZeroOneSeqIndividual

class DecimalMutation(Mutation):
	'''
	mutation operation for decimal encoded individuals:
	add random deviations(positive/negtive) at random positions of the selected individual
	'''
	def __init__(self, rate):
		'''
		mutation operation:
		rate: propability of mutation, [0,1]
		'''
		super().__init__(rate)

		# this operator is only available for DecimalIndividual
		self._individual_class = [DecimalFloatIndividual, DecimalIntegerIndividual]

	@staticmethod
	def mutate_individual(individual, positions, alpha, fun_evaluation=None):
		'''
		mutation method for decimal encoded individual:
		to add a random deviation for gene in specified positions
		- positions: 0-1 vector to specify positions for crossing
		- alpha: mutatation magnitude
		'''

		# for a gene G in range [L, U],
		# option 0: G = G + (U-G)*alpha
		# option 1:	G = G + (L-G)*alpha	

		# mutation options:
		# lower/upper bound
		L, U = individual.lb, individual.ub

		p = np.random.choice(np.arange(L, U+1), individual.dimension)
		t = np.zeros([8, 8])

		for i in range(6):
			t[0, i + 2] = p[i]
			t[1, i + 2] = p[i + 6]
		t[2, 4] = t[3, 4] = p[12]
		t[2, 5] = t[3, 5] = p[13]
		t[2, 6] = t[3, 6] = p[14]
		t[2, 7] = t[3, 7] = p[15]
		t[4, 6] = t[5, 6] = p[16]
		t[4, 7] = t[5, 7] = p[17]

		t = t * positions
		temp = np.copy(individual.solution)
		temp = temp * positions
		solution = individual.solution - temp + t

		
		# combine two mutation method
		# diff = ((U-individual.solution)-p*(U-L))*positions*alpha
		# solution = individual.solution +

		return solution.astype(np.int_)
		


class UniqueSeqMutation(Mutation):
	'''
	mutation operation for unique sequence individuals:
	exchange genes at random positions
	'''
	def __init__(self, rate):
		'''
		mutation operation:
		rate: propability of mutation, [0,1]
		'''
		super().__init__(rate)

		# this operator is only available for UniqueSeqIndividual
		self._individual_class = [UniqueSeqIndividual, UniqueLoopIndividual]


	@staticmethod
	def _mutate_positions(dimension):
		'''select random and continuous positions'''
		# start, end position
		pos = np.random.choice(dimension, 2, replace=False)
		start, end = pos.min(), pos.max()
		positions = np.zeros(dimension).astype(np.bool)
		positions[start:end+1] = True
		return positions

	@staticmethod
	def mutate_individual(individual, positions, alpha):
		'''
		reverse genes at specified positions:
		- positions: 0-1 vector to specify positions
		- alpha: probability to accept a worse solution
		'''
		solution = individual.solution.copy()		
		solution[positions] = solution[positions][::-1] # reverse genes at specified positions
		return solution