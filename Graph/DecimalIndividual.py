#----------------------------------------------------------
# Individual Object for GA
#----------------------------------------------------------
import numpy as np
from Individual import Individual


class DecimalFloatIndividual(Individual):
    '''
    decimal encoded individual, the solutions are float elements
    ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
    '''

    def init_solution(self, ranges):
        '''
        initialize random solution in `ranges`
        ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
        '''
        self._ranges = np.array(ranges)
        self._dimension = self._ranges.shape[0]

        # initialize solution within [lb, ub]
        seeds = np.random.random(self._dimension)
        lb = self._ranges[:, 0]
        ub = self._ranges[:, 1]
        self._solution = lb + (ub-lb)*seeds



class DecimalIntegerIndividual(Individual):
    '''
    dicimal encoded individual, the solutions are integer elements
    ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
    '''

    def init_solution(self, ranges):
        '''
        initialize random integer solution in `ranges`
        ranges: element range of solution, e.g. [(lb1, ub1), (lb2, ub2), ...]
        '''
        self._ranges = [8, 8]
        # self._dimension = np.array(ranges).shape[0]
        self._dimension = 18

        # initialize solution within [lb, ub]
        self.lb = 0
        self.ub = 5

        seeds = np.random.random(self._ranges)
        self._solution = np.triu(np.rint(self.lb + (self.ub-self.lb)*seeds).astype(int), k=1)
        self._solution[0, 1] = 0
        for i in range(3, self._ranges[0], 2):
            for j in range(i, self._ranges[0]):
                self._solution[i - 1, j] = self._solution[i, j]


    @property
    def solution(self):
        return self._solution

    @solution.setter
    def solution(self, solution):
        self._solution = np.rint(solution)
