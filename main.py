import sys
import pygame as pg
from time import sleep
from game import Snake, Apple

pg.init()

size = WIDTH, HEIGHT = 800, 800
screen = pg.display.set_mode(size)

board_size = 20
piece_size = WIDTH // board_size

snake = Snake(board_size, piece_size)

FRAME_RATE = 60
clock = pg.time.Clock()
dt: float = 0.0
counter: float = 0.0


def tick():
    snake.update()


while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()

        snake.handle_event(event)

    # updates
    if counter >= 0.125:
        tick()

        counter = 0

    # drawing
    screen.fill(pg.color.THECOLORS["black"])

    snake.draw(screen)

    pg.display.flip()
    dt = clock.tick(FRAME_RATE) / 1000.0
    counter += dt