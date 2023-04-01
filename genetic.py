import sys
from typing import List

import pygame as pg
from game import Snake, Apple, Board, MoveDirection
import random
from threading import Thread

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

N_ELITES: int = 10
N_MUTATION: int = 750
N_CROSSOVER: int = 750
N_POPULATION = N_ELITES + N_MUTATION + N_CROSSOVER
current_generation = 1

snakes: List[Snake] = [Snake(board_size, piece_size, True) for _ in range(N_POPULATION)]
population: List[NeuralNetwork] = []

for i in range(N_POPULATION):
    model = NeuralNetwork([
        InputLayer(28, linear),
        Layer(24, relu),
        Layer(12, relu),
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
    return snake.age ** 2 + (2 ** snake.apples_eaten)


def tick():
    snakes_alive = 0

    def task(ss: List[Snake], pp: List[NeuralNetwork]):
        for snake, model in zip(ss, pp):
            if not snake.alive:
                continue
            predictions = model.predict(snake.rays)
            snake.set_move_dir(MoveDirection(np.argmax(predictions)))
            snake.update()

    # threads = []
    # for i in range(10):
    #     size_s = len(snakes)//10
    #     thread = Thread(target=task, args=(snakes[i*size_s:(i+1)*size_s], population[i*size_s:(i+1)*size_s]))
    #     threads.append(thread)
    #     thread.start()
    #
    # for thread in threads:
    #     thread.join()
    #
    # anyAlive = False
    # for snake in snakes:
    #     if snake.alive:
    #         anyAlive = True
    #         break
    #
    # if not anyAlive:
    #     new_generation()

    for snake, model in zip(snakes, population):
        if not snake.alive:
            continue

        predictions = model.predict(snake.rays)
        snake.set_move_dir(MoveDirection(np.argmax(predictions)))
        snake.update()

        snakes_alive += 1

    if snakes_alive == 0:
        new_generation()


def new_generation():
    global population
    global snakes
    global current_generation

    fitnesses = [calculate_fitness(snake) for snake in snakes]
    sum_of_fitnesses = sum(fitnesses)

    probabilities = [fitness / sum_of_fitnesses for fitness in fitnesses]  # values between 0 and 1 - kinda unneccessary

    new_population = []
    population = [p for f, p in sorted(zip(probabilities, population), key=lambda pair: pair[0])]

    probabilities.sort()

    # mutations
    for i in range(N_MUTATION):
        rnd_indiv = get_random_individual_index(probabilities)
        new_indiv = population[rnd_indiv].copy()
        new_indiv.mutate(0.2)

        new_population.append(new_indiv)

    # crossovers

    for i in range(N_CROSSOVER):
        rnd_a = get_random_individual_index(probabilities)
        rnd_b = get_random_individual_index(probabilities)

        n_indiv = population[rnd_a].copy()
        n_indiv.crossover(population[rnd_b])
        new_population.append(n_indiv)

    # elitarism
    for individual in population[N_POPULATION - N_ELITES:]:
        new_population.append(individual.copy())

    population = new_population

    print("====== GENERATION REPORT ======")

    print(f"GENERATION: {current_generation}")
    print(f"AVG FITNESS: {sum_of_fitnesses / len(fitnesses)}")
    print(f"MAX FITNESS: {max(fitnesses)}")
    print(f"FITNESS TOTAL: {sum_of_fitnesses}")
    print(f"MAX APPLES: {max(snakes, key=lambda s: s.apples_eaten).apples_eaten}")

    print("===============================\n")

    for snake in snakes:
        snake.restart()

    current_generation += 1
    population[-1].save("brains/brain" + str(current_generation) + " " + str(max(fitnesses)) + ".npy")


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

    for snake in snakes[N_POPULATION - N_ELITES::N_ELITES // 10]:
        if snake.alive:
            snake.draw(screen)

    pg.display.flip()
    dt = clock.tick(FRAME_RATE) / 1000.0
    counter += dt
