import copy
import random
from collections import deque
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np


class GeneticProgramming:
    def __init__(
        self,
        terminals: List[str],
        functions: Dict[str, Callable],
        head_len: int = 20,
        total_genes: int = 4,
        max_arity: int = 2,
        tournament_size: int = 100,
        mutation_rate: float = 0.1,
        recombination_rate: float = 0.3,
        shift_rate: float = 0.1,
        shift_len: int = 3,
        size: int = 60,
        epochs: int = 50,
    ):
        self.functions = functions
        self.terminals = terminals
        self.function_symbols = list(functions.keys())
        self.elements = self.function_symbols + self.terminals
        self.total_genes = total_genes
        self.head_len = head_len
        self.tail_len = head_len * (max_arity - 1) + 1
        self.gene_len = self.head_len + self.tail_len
        self.tournament_size = tournament_size
        self.shift_len = shift_len
        self.mutation_rate = mutation_rate
        self.shift_rate = shift_rate
        self.recombination_rate = recombination_rate
        self.epochs = epochs
        self.size = size

    def generate_gene(self) -> List[str]:
        head = [random.choice(self.elements) for _ in range(self.head_len)]
        tail = [random.choice(self.terminals) for _ in range(self.tail_len)]
        return head + tail

    def init_population(self) -> List[List[List[str]]]:
        return [[self.generate_gene() for _ in range(self.total_genes)] for _ in range(self.size)]

    def eval(self, expression, value):
        try:
            queue = deque()
            args = 0
            for symbol in expression:
                if symbol in self.function_symbols:
                    args += 2
                queue.appendleft(symbol)
                if args == 0:
                    break
                else:
                    args -= 1
            exec_queue = deque()
            for symbol in list(queue):
                if symbol == "a":
                    exec_queue.append(value)
                if symbol in self.function_symbols:
                    args = 2
                    operand1 = exec_queue.popleft()
                    operand2 = exec_queue.popleft()
                    exec_queue.append(self.functions[symbol](operand2, operand1))
            return exec_queue.pop()
        except ZeroDivisionError:
            return 0

    def error(self, chromosome: List[List[str]], x: float, y: float) -> float:
        pred = sum([self.eval(''.join(gene), x) for gene in chromosome])
        return np.sqrt(np.power(pred - y, 2))

    def split_to_genes(self, chromo_str):
        chromosome = []
        for i in range(0, len(chromo_str), self.gene_len):
            start, end = i, i + self.gene_len
            chromosome.append(list(chromo_str[start:end]))
        return chromosome

    def join_chromosome(self, chromosome: List[List[str]]):
        return ''.join(''.join(gene) for gene in chromosome)

    def mutation(self, chromosome):
        if 0.1 < random.random():
            return chromosome
        gene_ = np.random.choice(np.arange(self.total_genes))
        new_gene = chromosome[gene_]
        new_sybmol = random.choice(self.elements)
        position = random.choice(np.arange(self.head_len))
        new_gene = new_gene[:position] + [new_sybmol] + new_gene[position + 1 :]
        chromosome[gene_] = new_gene
        return chromosome

    def one_point_recombination(self, chromo_a: List[List[str]], chromo_b: List[List[str]]):
        if self.recombination_rate < random.random():
            return chromo_a, chromo_b
        point = random.choice(np.arange(self.head_len))
        chromo_a = ''.join([''.join(gene) for gene in chromo_a])
        chromo_b = ''.join([''.join(gene) for gene in chromo_b])
        new_chromo_a = chromo_a[:point] + chromo_b[point:]
        new_chromo_b = chromo_b[:point] + chromo_a[point:]
        return self.split_to_genes(new_chromo_a), self.split_to_genes(new_chromo_b)

    def gene_recombination(self, chromo_a: List[List[str]], chromo_b: List[List[str]]):
        if self.recombination_rate < random.random():
            return chromo_a, chromo_b
        gene = random.choice(np.arange(self.total_genes))
        gene_a = copy.deepcopy(chromo_a[gene])
        chromo_a[gene] = chromo_b[gene][:]
        chromo_b[gene] = gene_a[:]
        return chromo_a, chromo_b

    def gene_shifting(self, chromosome):
        if self.shift_rate < random.random():
            return chromosome
        new_chromosome = copy.deepcopy(chromosome)
        gene = np.random.choice(np.arange(self.total_genes))
        selected_gene = new_chromosome.pop(gene)
        return [selected_gene] + new_chromosome

    def is_shifting(self, chromosome):
        if self.shift_rate < random.random():
            return chromosome
        new_chromosome = copy.deepcopy(chromosome)
        gene_a, gene_b = np.random.choice(self.total_genes, 2, replace=False)
        slice_point_a = np.random.choice(np.arange(self.gene_len - self.shift_len))
        slice_point_b = np.random.choice(np.arange(self.head_len - self.shift_len))
        gene_a_slice = new_chromosome[gene_a][slice_point_a : slice_point_a + self.shift_len]
        gene_b_head = new_chromosome[gene_b][: self.head_len]
        gene_b_head = gene_b_head[:slice_point_b] + gene_a_slice + gene_b_head[slice_point_b + self.shift_len : self.head_len]
        new_gene_b = gene_b_head + new_chromosome[gene_b][self.head_len :]
        new_chromosome[gene_b] = new_gene_b
        return new_chromosome

    def ris_shifting(self, chromosome):
        if self.shift_rate < random.random():
            return chromosome
        new_chromosome = copy.deepcopy(chromosome)
        gene = np.random.choice(self.total_genes)

        gene_head = new_chromosome[gene][: self.head_len]
        initial_slice_point = np.random.choice(self.head_len - self.shift_len)

        shift_len = self.shift_len
        slice_point = initial_slice_point
        for i, sybmol in enumerate(gene_head[initial_slice_point:]):
            if sybmol in self.function_symbols:
                break
            slice_point = initial_slice_point + i
            shift_len = self.shift_len - i

        if shift_len == 0:
            return chromosome

        gene_head = gene_head[slice_point : slice_point + shift_len] + gene_head[: self.head_len - shift_len]

        new_gene = gene_head + new_chromosome[gene][self.head_len :]
        new_chromosome[gene] = new_gene

        return new_chromosome

    def tournament_selection(self, population, train_data):
        candidates = [self.join_chromosome(chromosome) for chromosome in population]
        tournament = np.random.choice(candidates, size=self.tournament_size)
        tournament_fitnesses = [sum(self.error(self.split_to_genes(chromosome), x, y) for x, y in train_data) for chromosome in tournament]
        best_index = np.argmin(tournament_fitnesses)
        return self.split_to_genes(tournament[best_index])

    def predict(self, chromosome, x: float):
        return sum([self.eval(''.join(gene), x) for gene in chromosome])

    def solve(self, train_data: List[Tuple[float, float]]):
        population = self.init_population()
        best_fitness, best_chromo = None, None
        fitnesses = []
        for epoch in range(self.epochs):
            if epoch % 10 == 0 and fitnesses:
                print(f'Epoch: {epoch}, fitness: {min(fitnesses)}')
            new_population = []
            for i, chromosome in enumerate(population):
                chromo_a = self.mutation(chromosome)
                chromo_a = self.gene_shifting(chromo_a)
                chromo_a = self.is_shifting(chromo_a)
                chromo_a = self.ris_shifting(chromo_a)

                j = np.random.choice(np.delete(np.arange(self.size), i))
                chromo_b = population[j]

                chromo_a, chromo_b = self.one_point_recombination(chromo_a, chromo_b)
                # chromo_a, chromo_b = self.gene_recombination(chromo_a, chromo_b)

                new_population.append(chromo_a)
                new_population.append(chromo_b)

            population = [self.tournament_selection(new_population, train_data) for _ in range(self.size)]
            fitnesses = [sum(self.error(chromosome, x, y) for x, y in train_data) for chromosome in population]

            if best_fitness is None or min(fitnesses) < best_fitness:
                min_fitness_idx = np.argmin(fitnesses)
                best_fitness = fitnesses[min_fitness_idx]
                best_chromo = population[min_fitness_idx]

        return best_chromo
