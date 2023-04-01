import random

from game_genetic import GeneticSnake
from typing import List
import pygame as pg

from nn import NeuralNetwork
from nn.activation_functions import linear, relu, softmax
from nn.layer import InputLayer, Layer


class Population:
    snakes: List[GeneticSnake]

    N_ELITES: int = 10
    N_MUTATION: int = 250
    N_CROSSOVER: int = 250
    N_POPULATION = N_ELITES + N_MUTATION + N_CROSSOVER
    current_generation: int = 1

    # TODO: LOAD FROM settings.json

    def __init__(self, board_size: int, piece_size: int):

        self.board_size = board_size
        self.piece_size = piece_size

        self.snakes = []

        for i in range(self.N_POPULATION):
            model = NeuralNetwork([
                InputLayer(28, linear),
                Layer(24, relu),
                Layer(12, relu),
                Layer(4, softmax)
            ])

            snake = GeneticSnake(self.board_size, self.piece_size, model.copy())
            self.snakes.append(snake)

    def update(self):
        snakes_alive = 0

        for snake in self.snakes:
            if not snake.alive:
                continue

            snake.update()
            snakes_alive += 1

        if snakes_alive == 0:
            self.new_generation()

    def new_generation(self, verbose=True):

        for snake in self.snakes:
            snake.calculate_fitness()

        sum_of_fitnesses = sum(snake.fitness for snake in self.snakes)

        self.snakes.sort(key=lambda s: s.fitness)

        new_population: List[GeneticSnake] = []

        # mutations
        for i in range(self.N_MUTATION):
            rnd_indiv = self.get_random_individual(sum_of_fitnesses)
            new_indiv = GeneticSnake(self.board_size, self.piece_size, rnd_indiv.brain.copy())
            new_indiv.brain.mutate(0.2)
            new_population.append(new_indiv)

        # crossovers

        for i in range(self.N_CROSSOVER):
            rnd_a = self.get_random_individual(sum_of_fitnesses)
            rnd_b = self.get_random_individual(sum_of_fitnesses)

            n_indiv = GeneticSnake(self.board_size, self.piece_size, rnd_a.brain.copy())
            n_indiv.brain.crossover(rnd_b.brain)

            new_population.append(n_indiv)

        # elitarism
        for individual in self.snakes[self.N_POPULATION - self.N_ELITES:]:
            n_indiv = GeneticSnake(self.board_size, self.piece_size, individual.brain.copy())
            new_population.append(n_indiv)

        if verbose:
            self.print_generational_summary()

        self.snakes[-1].brain.save("brains/brain" + str(self.current_generation) + "_" + str(max(self.snakes, key=lambda x: x.fitness).fitness) + ".npy")

        self.snakes = new_population
        self.current_generation += 1


    def draw(self, screen: pg.Surface):
        for snake in self.snakes[self.N_POPULATION - self.N_ELITES::self.N_ELITES // 10]:
            if snake.alive:
                snake.draw(screen)

    def get_random_individual(self, sum_of_fitnesses) -> GeneticSnake:
        rnd = random.random() * sum_of_fitnesses

        for snake in self.snakes:
            if rnd - snake.fitness <= 0:
                return snake
            rnd -= snake.fitness

    def print_generational_summary(self):
        print("====== GENERATION REPORT ======")

        print(f"GENERATION: {self.current_generation}")
        print(f"MIN FITNESS: {min(self.snakes, key=lambda s: s.fitness).fitness}")
        print(f"MAX FITNESS: {max(self.snakes, key=lambda s: s.fitness).fitness}")
        print(f"MAX APPLES: {max(self.snakes, key=lambda s: s.apples_eaten).apples_eaten}")
        print(f"MIN APPLES: {min(self.snakes, key=lambda s: s.apples_eaten).apples_eaten}")

        print("===============================\n")
