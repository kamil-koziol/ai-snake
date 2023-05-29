import sys

sys.path.append('../ai-snake')

import pygame as pg
from game import Snake, Board, MoveDirection
from typing import List, Tuple
from astar import AStar

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
DELAY = 0.05

board = Board(board_size, piece_size)

def tick():
    astar: AStar = AStar((board_size, board_size))

    def vector_to_tuple(v: pg.Vector2) -> Tuple[int, int]:
        return int(v.x), int(v.y)

    for snake_piece in snake.pieces:
        astar.set_wall(vector_to_tuple(snake_piece))

    path = astar.find_path(vector_to_tuple(snake.pos), vector_to_tuple(snake.apple.pos))
    if not path:
        return
        # path = astar.find_path(vector_to_tuple(snake.pos), vector_to_tuple(snake.pieces[-1]), longest=True)

    next_step = list(path[1])
    next_step[0] -= snake.pos.x
    next_step[1] -= snake.pos.y

    move_dir: MoveDirection

    if next_step[0] == 1:
        move_dir = MoveDirection.RIGHT
    elif next_step[0] == -1:
        move_dir = MoveDirection.LEFT
    elif next_step[1] == 1:
        move_dir = MoveDirection.DOWN
    elif next_step[1] == -1:
        move_dir = MoveDirection.UP

    snake.set_move_dir(move_dir)
    snake.update()


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
