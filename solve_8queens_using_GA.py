import math, random
import pandas as pd
import numpy as np
from queue import SimpleQueue

PROBLEM_SIZE = 8
initialPopulationSize = 20
# CROSSOVER_POINT = 0.5
MUTATION_RATE = 0.3
HALT = 1000

targetFitness = math.comb(PROBLEM_SIZE, 2)			# max number of non-attacking pairs - 8C2
print('targetFitness:', targetFitness)

gene_list = [x for x in range(PROBLEM_SIZE)] 
print('gene_list:', gene_list)


def getFitnessScore(chromosome):
	maxFitness = math.comb(PROBLEM_SIZE, 2)

	# no horizontal or vertical conflicts beacause of representation choice
	# calculate number of diagonal conflicts
	dConflict = 0
	for i in range(len(chromosome)):
		for j in range(len(chromosome)):
			if (i != j) and (abs(chromosome[i] - chromosome[j]) == abs(i - j)):
				dConflict += 1
	dConflict = dConflict // 2	

	return (maxFitness - dConflict)


def getProbability(fitness):
	return (fitness / targetFitness)


def reproduce(chromosome1, chromosome2):
	# crossover_index = int(CROSSOVER_POINT * len(chromosome1))
	# child1 = chromosome1[0:crossover_index] + chromosome2[crossover_index:]
	# child2 = chromosome2[0:crossover_index] + chromosome1[crossover_index:]
	# return child1, child2

	pos1, pos2 = random.sample([x for x in range(1, len(chromosome1) - 1)], k=2)
	if pos1 > pos2:
		pos1, pos2 = pos2, pos1

	genesFrom1 = chromosome1[pos1:pos2+1]
	genesFrom2 = SimpleQueue()
	for i in range(len(chromosome2)):
		gene = chromosome2[i]
		if gene not in genesFrom1:
			genesFrom2.put(gene)
	
	child = []
	for i in range(pos1):
		child.append(genesFrom2.get())

	child = child + genesFrom1

	for i in range(len(child), len(chromosome2)):
		child.append(genesFrom2.get())
	
	return child


def mutate(chromosome):
	if random.random() > MUTATION_RATE:
		pos1, pos2 = random.sample([x for x in range(0, len(chromosome))], k=2)
		chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]

	return chromosome

##-------------------------------------------------------------------------------------------------

# Create initial population
populationList = [random.sample(gene_list, PROBLEM_SIZE) for x in range(initialPopulationSize)]
populationDF = pd.DataFrame()
populationDF['Chromosome'] = populationList
populationDF['FitnessScore'] = populationDF['Chromosome'].apply(getFitnessScore)
populationDF['SurvivalProbability'] = populationDF['FitnessScore'].apply(getProbability)

print('Initial population:', populationDF.head())

generation = 1
print ('Trying with max reproduction cycles of', HALT)
while not targetFitness in populationDF['FitnessScore'].tolist():
	if (generation % 10 == 0):
		print('Generation ', generation)

	# Select two random candidates with bias towards more fit ones
	index1, index2 = random.sample([x for x in range(populationDF.shape[0] // 2)], 2)
	parent1 = populationDF.at[index1, 'Chromosome']
	parent2 = populationDF.at[index2, 'Chromosome']
	# print('parents:', parent1, 'x', parent2)

	# Create offsprings from crossover of the parents
	# child1, child2 = reproduce(parent1, parent2)
	child1 = reproduce(parent1, parent2)
	child2 = reproduce(parent2, parent1)

	# Introduce mutaion in the offsprings
	child1, child2 = mutate(child1), mutate(child2)

	# Add offsprings to the populations
	populationDF = populationDF.append({'Chromosome': child1, 'FitnessScore': getFitnessScore(child1), 'SurvivalProbability': getProbability(getFitnessScore(child1))}, ignore_index=True)
	populationDF = populationDF.append({'Chromosome': child2, 'FitnessScore': getFitnessScore(child2), 'SurvivalProbability': getProbability(getFitnessScore(child2))}, ignore_index=True)

	# Order the population based on survival probability in order to identify candidates with good genes
	populationDF = populationDF.sort_values(['SurvivalProbability'], ascending=False).reset_index(drop=True)	# rank based on survival chance
	populationDF = populationDF[:-2]		# kill two chromosomes least likely to survive
	# print(populationDF.head())

	if generation == HALT:
		break

	generation += 1

# print('Population:', populationDF)

if targetFitness in populationDF['FitnessScore'].tolist():
	print('Found solution after generation', generation)
	solutionDF = populationDF.loc[populationDF['FitnessScore'] == targetFitness]
	print(solutionDF)
else:
	print('No solution found upto generation ', generation, ', stopping here.')
	
