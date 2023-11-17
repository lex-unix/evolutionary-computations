import random
from typing import List

import numpy as np
from numpy.random import randint

from src.lib import Individ
from src.lib.common import calculate_similarity


def panmixia(population: List[Individ]):
    return [random.choice(population), random.choice(population)]


def selection(population: List[Individ]):
    mean = np.mean(np.round([individ.fitness for individ in population], 4))
    candidates = [individ for individ in population if individ.fitness <= mean]
    if not candidates:
        min_fitness = min([individ.fitness for individ in population])
        candidates = [
            individ for individ in population if individ.fitness == min_fitness
        ]
    return [random.choice(candidates), random.choice(candidates)]


def inbreeding(population: List[Individ]):
    pos = randint(len(population))
    parent_a = population[pos]
    min_similarity = float('inf')
    selected_parents = []
    for i in range(len(population)):
        if i == pos:
            continue
        similarity = calculate_similarity(parent_a, population[i])
        if similarity < min_similarity:
            min_similarity = similarity
            selected_parents = [parent_a, population[i]]

    return selected_parents


def outbreeding(population: List[Individ]):
    pos = randint(len(population))
    parent_a = population[pos]
    max_dissimilarity = -1
    selected_parents = None

    for i in range(len(population)):
        if i == pos:
            continue
        dissimilarity = calculate_similarity(parent_a, population[i])
        if dissimilarity > max_dissimilarity:
            max_dissimilarity = dissimilarity
            selected_parents = [population[i], population[i]]

    return selected_parents


def crossover(parent_a: Individ, parent_b: Individ) -> List[Individ]:
    point = randint(1, len(parent_a.genotype))

    chromo_a = parent_a.genotype[:point] + parent_b.genotype[point:]
    chromo_b = parent_b.genotype[:point] + parent_a.genotype[point:]

    child_a = Individ(genotype=chromo_a)
    child_b = Individ(genotype=chromo_b)

    return [child_a, child_b]


def two_point_crossover(parent_a: Individ, parent_b: Individ) -> List[Individ]:
    genotype_length = len(parent_a.genotype)

    cut_point_1, cut_point_2 = sorted(random.sample(range(genotype_length), 2))

    child_1_genotype = (
        parent_a.genotype[:cut_point_1] +
        parent_b.genotype[cut_point_1:cut_point_2] +
        parent_a.genotype[cut_point_2:]
    )

    child_2_genotype = (
        parent_b.genotype[:cut_point_1] +
        parent_a.genotype[cut_point_1:cut_point_2] +
        parent_b.genotype[cut_point_2:]
    )

    child_a = Individ(genotype=child_1_genotype)
    child_b = Individ(genotype=child_2_genotype)

    return [child_a, child_b]


def elitism(parents: List[Individ], children: List[Individ]) -> List[Individ]:
    combined = parents + children
    combined_sorted = sorted(combined, key=lambda x: x.fitness)
    return combined_sorted[:len(parents)]


def simple(children: List[Individ], _) -> List[Individ]:
    return children


def select_with_displacement(parents: List[Individ], children: List[Individ], num_elites: int = 10) -> List[Individ]:
    new_population = []
    for child in children:
        if child not in parents:
            new_population.append(child)
    if len(new_population) < len(parents):
        diff = len(parents) - len(new_population)
        sorted_parents = sorted(parents, key=lambda x: x.fitness)
        new_population.extend(sorted_parents[:diff])
    return new_population
