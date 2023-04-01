import sys
from typing import List

import pygame as pg
from game import Snake, Apple, Board, MoveDirection
from game_genetic import Population

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
population = Population(board_size, piece_size)

def tick():
    population.update()


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
    population.draw(screen)


    pg.display.flip()
    dt = clock.tick(FRAME_RATE) / 1000.0
    counter += dt
