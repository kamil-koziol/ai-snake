import sys
from typing import List

import pygame as pg
from game import Snake, Apple, Board, MoveDirection
import random

import numpy as np

from nn import NeuralNetwork, Layer
from nn.activation_functions import relu, softmax, linear
from nn.layer import InputLayer

pg.init()

size = WIDTH, HEIGHT = 800, 800
screen = pg.display.set_mode(size)

board_size = 20
piece_size = WIDTH // board_size

FRAME_RATE = 60
clock = pg.time.Clock()
dt: float = 0.0
counter: float = 0.0
DELAY = 0.01

board = Board(board_size, piece_size)

# Genetic parameters

N_ELITES: int = 30
N_MUTATION: int = 1000
N_CROSSOVER: int = 0
N_POPULATION = N_ELITES + N_MUTATION + N_CROSSOVER

snakes: List[Snake] = [Snake(board_size, piece_size, True) for _ in range(N_POPULATION)]

population = []
for i in range(N_POPULATION):
    model = NeuralNetwork([
        InputLayer(24),
        Layer(32, relu),
        Layer(16, relu),
        Layer(4, softmax)
    ])

    population.append(model)


def get_random_individual_index(fitnesses):
    rnd = random.random()
    for i, f in enumerate(fitnesses):
        if rnd - f <= 0:
            return i
        rnd -= f


def calculate_fitness(snake: Snake) -> float:
    return (snake.apples_eaten+1) ** 2

def tick():
    snakes_alive = 0
    for snake, model in zip(snakes, population):
        if not snake.alive:
            continue

        snake.update()
        predictions = model.predict(snake.rays)
        snake.set_move_dir(MoveDirection(np.argmax(predictions)))

        snakes_alive += 1


    if snakes_alive == 0:
        new_generation()


def new_generation():
    global population
    global snakes

    fitnesses = [calculate_fitness(snake) for snake in snakes]
    sum_of_fitnesses = sum(fitnesses)

    fitnesses = [fitness/sum_of_fitnesses for fitness in fitnesses] # values between 0 and 1 - kinda unneccessary

    new_population = []
    population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
    fitnesses.sort()

    # mutations
    for i in range(N_MUTATION):
        rnd_indiv = get_random_individual_index(fitnesses)
        new_indiv = population[rnd_indiv].copy()
        new_indiv.mutate(0.3)

        new_population.append(new_indiv)

    # crossovers

    for i in range(N_CROSSOVER):
        rnd_a = get_random_individual_index(fitnesses)
        rnd_b = get_random_individual_index(fitnesses)

    # elitarism
    for individual in population[N_POPULATION - N_ELITES:]:
        new_population.append(individual.copy())

    del population
    population = new_population
    for snake in snakes:
        snake.restart()


longest_snake = snakes[0]

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()

    # updates
    if counter >= DELAY:
        tick()

        counter = 0

    # drawing
    screen.fill(pg.color.THECOLORS["black"])

    board.draw(screen)

    tmp_longest = max(snakes, key=lambda s: s.apples_eaten)
    if tmp_longest.apples_eaten > longest_snake.apples_eaten:
        longest_snake = tmp_longest

    longest_snake.draw(screen)
    # for snake in snakes[::-1]:
    #     if snake.alive:
    #         snake.draw(screen)
    #         break

    pg.display.flip()
    dt = clock.tick(FRAME_RATE) / 1000.0
    counter += dt
