import sys

import pygame
import pygame as pg
from time import sleep
from game import Snake, MoveDirection, Board

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
DELAY = 0.01

board = Board(board_size, piece_size)

def tick():
    snake.update()

    if snake.pos.x == board_size - 1 and snake.move_dir != MoveDirection.UP:
        if snake.move_dir == MoveDirection.RIGHT:
            snake.move_dir = MoveDirection.DOWN

        elif snake.move_dir == MoveDirection.DOWN:
            snake.move_dir = MoveDirection.LEFT

    if snake.pos.x == 1 and snake.pos.y != board_size - 1:
        if snake.move_dir == MoveDirection.LEFT:
            snake.move_dir = MoveDirection.DOWN

        elif snake.move_dir == MoveDirection.DOWN:
            snake.move_dir = MoveDirection.RIGHT

    if snake.pos.x == 0 and snake.pos.y == board_size - 1:
        snake.move_dir = MoveDirection.UP

    if snake.pos.x == 0 and snake.pos.y == 0:
        snake.move_dir = MoveDirection.RIGHT

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

    snake.draw(screen)


    pg.display.flip()
    dt = clock.tick(FRAME_RATE) / 1000.0
    counter += dt
