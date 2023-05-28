import random
import sys
from typing import List
import os

import neat
import numpy as np

import pygame as pg
from game import Snake, Apple, Board, MoveDirection
from neat_snake import NEATSnake

best_seed = 0
best_apples = 0

pg.init()

size = WIDTH, HEIGHT = 800, 800
screen = pg.display.set_mode(size)

board_size = 50
piece_size = WIDTH // board_size

FRAME_RATE = 60

board = Board(board_size, piece_size)


def main(genomes, config):
    ge = []
    snakes: List[NEATSnake] = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        snakes.append(NEATSnake(board.board_size, board.piece_size, net, genome_id))

        genome.fitness = 0
        ge.append(genome)

    clock = pg.time.Clock()
    dt: float = 0.0
    counter: float = 0.0
    DELAY = 0.01

    amount_of_seeds = 5

    for seed in range(amount_of_seeds):

        np.random.seed(seed)
        random.seed(seed)
        print(f"SEED = {seed}")

        while True:
            # updates
            if counter >= DELAY:
                amount_of_alive_snakes = 0
                for i, snake in enumerate(snakes):

                    if not snake.alive:
                        continue

                    # deciding move direction for snake
                    snake.update()

                    amount_of_alive_snakes += 1

                counter = 0

                if amount_of_alive_snakes == 0:
                    for i in range(len(snakes)):
                        snakes[i].calculate_fitness()
                        if snakes[i].fitness > ge[i].fitness:
                            ge[i].fitness = snakes[i].fitness

                        snakes[i].restart()
                    break

            # drawing
            screen.fill(pg.color.THECOLORS["black"])

            board.draw(screen)

            longest_snakes = reversed(sorted(snakes, key=lambda s: s.apples_eaten))

            printed_snakes = 0
            for i, snake in enumerate(longest_snakes):
                if not snake.alive:
                    continue

                snake.draw(screen)
                printed_snakes += 1

                if printed_snakes >= 3:
                    break

            pg.display.flip()
            dt = clock.tick(FRAME_RATE) / 1000.0
            counter += dt


def run(config_file: str):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix = "checkpoints2/neat-checkpoint-"))

    # p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-1999')
    # Run for up to 300 generations.
    winner = p.run(main, 2000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

def main_best(genomes, config):
    ge = []
    snakes: List[NEATSnake] = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        if genome_id == 761357:
            snakes.append(NEATSnake(board.board_size, board.piece_size, net, genome_id))

            genome.fitness = 0
            ge.append(genome)

    clock = pg.time.Clock()
    dt: float = 0.0
    counter: float = 0.0
    DELAY = 0.00

    while True:
        # updates

        if counter >= DELAY:
            amount_of_alive_snakes = 0
            for i, snake in enumerate(snakes):

                if not snake.alive:
                    continue

                # deciding move direction for snake
                snake.update()

                amount_of_alive_snakes += 1

            counter = 0

            if amount_of_alive_snakes == 0:
                for i in range(len(snakes)):
                    snakes[i].calculate_fitness()
                    ge[i].fitness = snakes[i].fitness

                with open("seeds.txt", 'a') as f:
                    f.write(f"apples = {snakes[0].apples_eaten}\n")
                print(snakes[0].apples_eaten)
                return

        # drawing
        screen.fill(pg.color.THECOLORS["black"])

        board.draw(screen)

        for snake in snakes:
            snake.draw(screen)

        pg.display.flip()
        dt = clock.tick(FRAME_RATE) / 1000.0
        counter += dt


def start(p, seed):
    print(f"seed = {seed}, ", end="")

    with open("seeds.txt", 'a') as f:
        f.write(f"seed = {seed}, ")
    np.random.seed(seed)
    random.seed(seed)
    try:
        winner = p.run(main_best, 1)
    except:
        print("")

def run_best_individual(checkpoint, species_id,  genome_id, config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Checkpointer.restore_checkpoint(checkpoint)

    for seed in range(10000):
        start(p, seed)
    # print(winner)
    # main_best(winner, config)


if __name__ == "__main__":
    pg.init()
    screen = pg.display.set_mode(size)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    run(config_path) # for training
    # run_best_individual("checkpoints/neat-checkpoint-1677", 706, 758141, config_path)
