from __future__ import annotations

import random

import pygame
import pygame as pg
from typing import List, Optional
from enum import Enum

from game import Apple, Block
import numpy as np
import pandas as pd


class MoveDirection(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class RaysDirections(Enum):
    UP = 0
    UP_RIGHT = 1
    RIGHT = 2
    DOWN_RIGHT = 3
    DOWN = 4
    DOWN_LEFT = 5
    LEFT = 6
    UP_LEFT = 7


class Snake:
    pos: pg.Vector2
    pieces: List[pg.Vector2]
    piece_size: int
    move_dir: MoveDirection
    apples_eaten: int
    age: int
    hunger: int

    apple: Apple

    alive: bool

    rays: np.ndarray[float]

    DEFAULT_HUNGER = 200
    STARTING_SNAKE_SIZE = 3

    def __init__(self, board_size, piece_size, hunger_enabled=False, print_to_file=False):
        self.print_to_file = print_to_file
        self.piece_size = piece_size
        self.board_size = board_size
        self.hunger_enabled = hunger_enabled

        self.setup()

    def setup(self):
        self.pos = pg.Vector2(random.randint(0, self.board_size - 1 - self.STARTING_SNAKE_SIZE),
                              random.randint(0, self.board_size - 1 - self.STARTING_SNAKE_SIZE))
        self.pieces = []
        self.pieces.append(self.pos.copy())

        self.move_dir = MoveDirection(random.randint(0, len(MoveDirection) - 1))

        self.initial_tail_setup(self.move_dir, self.STARTING_SNAKE_SIZE)

        self.apples_eaten = 0
        self.age = 0
        self.set_new_apple()
        self.rays = np.zeros((1, 28))
        self.alive = True
        self.hunger = Snake.DEFAULT_HUNGER
        self.update_rays()

    def set_new_apple(self):
        apple = Apple(self.board_size, self.piece_size)
        apple.set_to_random_position(self.pieces)
        self.apple = apple

    def handle_apple_collision(self):
        if self.pos == self.apple.pos:
            self.on_apple_eat()

    def on_apple_eat(self):
        self.grow()
        self.apples_eaten += 1
        self.apple.set_to_random_position(self.pieces)
        self.hunger = self.DEFAULT_HUNGER

    def update(self, verbose=0):
        if not self.alive:
            return

        self.pos += self.get_dir_vector(self.move_dir)
        self.pieces_update()

        self.handle_walls()
        self.handle_self_collision()
        self.handle_apple_collision()
        self.handle_hunger()

        self.update_rays()
        # self.update_rays_binary()
        self.age += 1

        if verbose == 1:
            print("=" * 8)
            print(self)

            with np.printoptions(precision=2, suppress=True):
                print("WALL", self.rays[0, 0: 8])
                print("APPL", self.rays[0, 8: 16])
                print("SELF", self.rays[0, 16: 24])
                print("DIRS", self.rays[0, 24:])

            print("=" * 8)
            print()

    def pieces_update(self):
        self.pieces.pop()
        self.pieces.insert(0, self.pos.copy())

    def initial_tail_setup(self, movedir: MoveDirection, size):
        # creating initial lenght
        mdir = self.get_dir_vector(movedir) * -1
        npos = self.pos.copy()
        for i in range(size - 1):
            npos += mdir
            self.pieces.append(npos.copy())

    def grow(self):
        self.pieces.append(self.pieces[-1])

    def get_dir_vector(self, dir: MoveDirection):
        if dir == MoveDirection.UP:
            return pg.Vector2(0, -1)
        elif dir == MoveDirection.RIGHT:
            return pg.Vector2(1, 0)
        elif dir == MoveDirection.DOWN:
            return pg.Vector2(0, 1)
        elif dir == MoveDirection.LEFT:
            return pg.Vector2(-1, 0)

    def draw(self, screen: pg.Surface):
        for piece in self.pieces:
            b = Block(piece, self.piece_size, pg.Color(50, 230, 0))
            b.draw(screen)

        self.apple.draw(screen)

    def print_data(self, move_dir: MoveDirection):

        if self.print_to_file:

            data =self.rays[0]
            data_with_move_dir = np.append(data, move_dir.value)

            df = pd.DataFrame( np.expand_dims(data_with_move_dir, axis=0))
            df.to_csv('data.csv', mode='a', sep=';', index=False, header=False)

    def handle_event(self, event: pg.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.set_move_dir(MoveDirection.LEFT)
            elif event.key == pygame.K_RIGHT:
                self.set_move_dir(MoveDirection.RIGHT)
            elif event.key == pygame.K_DOWN:
                self.set_move_dir(MoveDirection.DOWN)
            if event.key == pygame.K_UP:
                self.set_move_dir(MoveDirection.UP)

    def handle_walls(self):
        if self.pos.x < 0 or self.pos.x > self.board_size - 1 or self.pos.y < 0 or self.pos.y > self.board_size - 1:
            self.die()

    def die(self):
        self.alive = False

    def update_rays(self):

        # TODO: MAKE RELATIVE RAYS TO CURRENT DIRECTION

        DIAG = 1.40
        MAX_DISTANCE = self.board_size * DIAG
        # walls

        self.rays[0, RaysDirections.UP.value] = (self.pos.y)
        self.rays[0, RaysDirections.RIGHT.value] = (self.board_size - self.pos.x - 1)
        self.rays[0, RaysDirections.DOWN.value] = (self.board_size - self.pos.y - 1)
        self.rays[0, RaysDirections.LEFT.value] = (self.pos.x)

        self.rays[0, RaysDirections.UP_RIGHT.value] = min(self.rays[0, RaysDirections.UP.value],
                                                          self.rays[0, RaysDirections.RIGHT.value]) * DIAG
        self.rays[0, RaysDirections.DOWN_RIGHT.value] = min(self.rays[0, RaysDirections.DOWN.value],
                                                            self.rays[0, RaysDirections.RIGHT.value]) * DIAG
        self.rays[0, RaysDirections.DOWN_LEFT.value] = min(self.rays[0, RaysDirections.DOWN.value],
                                                           self.rays[0, RaysDirections.LEFT.value]) * DIAG
        self.rays[0, RaysDirections.UP_LEFT.value] = min(self.rays[0, RaysDirections.UP.value],
                                                         self.rays[0, RaysDirections.LEFT.value]) * DIAG

        self.rays[0, RaysDirections.UP.value::2] /= self.board_size
        self.rays[0, RaysDirections.UP_RIGHT.value::2] /= MAX_DISTANCE
        self.rays[0, 0:8] = 1 - self.rays[0, 0:8]

        # apple

        apple_dir = self.get_direction_between_vectors(self.pos, self.apple.pos)
        diff = (self.apple.pos - self.pos)

        for direction in range(len(RaysDirections)):
            self.rays[0, len(RaysDirections) + direction] = MAX_DISTANCE

        if apple_dir:
            self.rays[0, len(RaysDirections) + apple_dir.value] = diff.magnitude()

        self.rays[0, 8:16] /= MAX_DISTANCE
        self.rays[0, 8:16] = 1 - self.rays[0, 8:16]

        # self

        smallest = np.repeat(MAX_DISTANCE, len(RaysDirections))
        for piece in self.pieces[1:]:
            piece_dir = self.get_direction_between_vectors(self.pos, piece)
            if piece_dir:
                dist = (piece - self.pos).magnitude()
                if dist < smallest[piece_dir.value]:
                    smallest[piece_dir.value] = dist

        for direction in range(len(RaysDirections)):
            self.rays[0, len(RaysDirections) * 2 + direction] = smallest[direction]

        self.rays[0, 16:24] /= MAX_DISTANCE
        self.rays[0, 16:24] = 1 - self.rays[0, 16:24]

        self.rays[0, 24] = 1 if self.move_dir == MoveDirection.UP else 0
        self.rays[0, 25] = 1 if self.move_dir == MoveDirection.RIGHT else 0
        self.rays[0, 26] = 1 if self.move_dir == MoveDirection.DOWN else 0
        self.rays[0, 27] = 1 if self.move_dir == MoveDirection.LEFT else 0

        # rotate rays to relative direction

        # movement = -self.move_dir.value*2
        #
        # for i in range(3):
        #     tmp_rays = np.zeros(len(RaysDirections))
        #
        #     for j in range(len(RaysDirections)):
        #         tmp_rays[(j + movement) % len(RaysDirections)] = self.rays[0, i*len(RaysDirections) + j]
        #
        #     for j in range(len(RaysDirections)):
        #         self.rays[0, i * len(RaysDirections) + j] = tmp_rays[j]

    def update_rays_binary(self):

        DIAG = 1.4
        MAX_DISTANCE = self.board_size * DIAG

        self.update_rays()
        for i in range(len(RaysDirections)):
            smallest = 1
            for j in range(3):
                if self.rays[0, len(RaysDirections) * j + i] <= smallest:
                    smallest = self.rays[0, len(RaysDirections) * j + i]

            for j in range(3):
                if self.rays[0, len(RaysDirections) * j + i] == smallest:
                    self.rays[0, len(RaysDirections) * j + i] = 1
                else:
                    self.rays[0, len(RaysDirections) * j + i] = 0

    def get_direction_between_vectors(self, vec1: pg.Vector2, vec2: pg.Vector2) -> Optional[RaysDirections]:
        diff = (vec2 - vec1)
        if diff.x == 0 and diff.y == 0:
            return None

        diffn = diff.normalize()

        if diffn.x == 1 and diffn.y == 0:
            return RaysDirections.RIGHT

        elif diffn.x == -1 and diffn.y == 0:
            return RaysDirections.LEFT

        elif diffn.x == 0 and diffn.y == -1:
            return RaysDirections.UP

        elif diffn.x == 0 and diffn.y == 1:
            return RaysDirections.DOWN

        elif abs(diffn.x) == abs(diffn.y):
            # DIAGONALS
            if diffn.x == diffn.y and diffn.x > 0:
                return RaysDirections.DOWN_RIGHT
            elif diffn.x == diffn.y and diffn.x < 0:
                return RaysDirections.UP_LEFT
            elif diffn.x > diffn.y:
                return RaysDirections.UP_RIGHT
            elif diffn.x < diffn.y:
                return RaysDirections.DOWN_LEFT

        return None

    def handle_self_collision(self):
        if self.pos in self.pieces[1:]:
            self.die()

    def handle_hunger(self):
        self.hunger -= 1
        if self.hunger == 0:
            self.die()

    def set_move_dir(self, move_dir: MoveDirection):

        if self.move_dir == MoveDirection.DOWN and move_dir == MoveDirection.UP:
            return
        if self.move_dir == MoveDirection.UP and move_dir == MoveDirection.DOWN:
            return
        if self.move_dir == MoveDirection.LEFT and move_dir == MoveDirection.RIGHT:
            return
        if self.move_dir == MoveDirection.RIGHT and move_dir == MoveDirection.LEFT:
            return

        self.print_data(move_dir)
        self.move_dir = move_dir

    def restart(self):
        self.setup()
