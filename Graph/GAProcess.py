#----------------------------------------------------------
# Simple Genetic Algorithm
#----------------------------------------------------------
import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

# from train_cora import Model
from train_ogb import Model

def learn_model(history, valid = True):

	X = np.array([i.solution for i in history])
	X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
	y = np.array([i.fitness for i in history])
	if valid:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
		rf_regressor.fit(X_train, y_train)
		y_pred = rf_regressor.predict(X_test)
		r2 = r2_score(y_test, y_pred)
		mse = mean_squared_error(y_test, y_pred)
		print('R2 score: {:.2f}'.format(r2))
		print('MSE: {:.2f}'.format(mse))
	rf_regressor.fit(X, y)
	return rf_regressor.predict


class GA():
	'''Simple Genetic Algorithm'''
	def __init__(self, population, selection, crossover, mutation, fun_fitness=None):
		'''
		fun_fitness: fitness based on objective values. minimize the objective by default
		'''		
		# check compatibility between Individual and GA operators
		if not crossover.individual_class or not population.individual.__class__ in crossover.individual_class:
			raise ValueError('incompatible Individual class and Crossover operator')  
		if not mutation.individual_class or not population.individual.__class__ in mutation.individual_class:
			raise ValueError('incompatible Individual class and Mutation operator')

		self.population = population
		self.selection = selection
		self.crossover = crossover
		self.mutation = mutation
		self.fun_fitness = fun_fitness if fun_fitness else (lambda x:np.arctan(-x)+np.pi)

	def run(self, gen=50):
		'''
		solve the problem based on Simple GA process
		two improved methods could be considered:
			a) elitism mechanism: keep the best individual, i.e. skip the selection, crossover, mutation operations
			b) adaptive mechenism: adaptive crossover rate, adaptive mutation megnitude. 
		'''

		# initialize population
		self.population.initialize()
		history = []
		children = []
		spear_correlation = -100
		represent_number = 4
		mutation_times = 1

		arch_matrices = [i.solution for i in self.population.individuals]

		# training the real model to get the valid accuracy as the fitness
		valid_acc = Model(arch_matrices)
		for I, f in zip(self.population.individuals, valid_acc):
			I.evaluation = f
			I.fitness = f
			history.append(I)

		# solving process
		flag = 0
		for n in range(1, gen+1):
			print('run:', n)
			# update the predictor by training the MODEL
			MODEL = learn_model(history)
      
			# evaluate and get the best individual in previous generation
			# self.population.evaluate(predictor, self.fun_fitness) #xiezaiqianmian?
			# self.population.evaluate(fun_evaluation, self.fun_fitness)
			# the_best = copy.deepcopy(self.population.best)

			# selection
			# self.population.individuals = self.selection.select(self.population)
			parents = self.selection.select(self.population)
			from Population import Population
			parents_pop = Population(individual=parents[0], size=len(parents))
			parents_pop.individuals = parents

			# crossover
      		# self.population.individuals = self.crossover.cross(self.population)
			child = self.crossover.cross(parents_pop, n)

			# mutation
			for i in range(mutation_times):
				rate = 1.0 - (0.1 + 0.4 * np.random.rand() * n / gen) # 0.5~0.9
				self.mutation.mutate(child, rate)

				# set attributes for each individual
				for I in child:
					if I.evaluation is None:
						I.evaluation = MODEL(I.solution.reshape(1, -1))
						I.fitness = I.evaluation
					children.append(I)

			# select representative, temporally ignore
			slt_children = sorted(children, key=lambda x: x.fitness, reverse=True)
			import math
			slt_children = slt_children[0::math.floor(len(slt_children)/represent_number)][:represent_number]

			arch_matrices = [i.solution for i in slt_children]
			valid_acc = Model(arch_matrices)
			spear_x = []
			spear_y = []
			for I, f in zip(slt_children, valid_acc):
				spear_x.append(I.fitness)
				spear_y.append(f)
				I.evaluation = f
				I.fitness = f
				history.append(I)
			temp_correlation, p_value = stats.spearmanr(spear_x, spear_y)
			with open("spearman_hiv.txt", "a+") as f:
				print('temp_correlation:', temp_correlation, file=f)
      
			if temp_correlation > spear_correlation:
				mutation_times += 1
				spear_correlation = temp_correlation
				if represent_number > 4:
					represent_number -= 1

			else:
				flag += 1
				if flag == 2:
					represent_number += 1
					flag = 0



			# elitism mechanism: 
			# set a random individual as the best in previous generation
			evaluation_child = np.array([I.evaluation for I in children])
			pos_best = np.argmax(evaluation_child)
			evaluation_population = np.array([I.evaluation for I in self.population.individuals])
			pos_worst = np.argmin(evaluation_population)
      
			children[pos_best].evaluation = Model(copy.deepcopy(children[pos_best].solution[np.newaxis, :]))[0]
			self.population.individuals[pos_worst] = copy.deepcopy(children[pos_best])
			if n % 1 == 0:
				with open('result_hiv.txt','a+') as GA_result:
					print('running time: {0}, GA solution input: {1}'.format(n, self.population.best.solution), file=GA_result)
					print('running time: {0}, GA solution output: {1}'.format(n, self.population.best.evaluation), file=GA_result)

		self.population.evaluate(MODEL, self.fun_fitness)
		return self.population.best
