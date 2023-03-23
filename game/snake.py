from __future__ import annotations

import pygame
import pygame as pg
from typing import List, Optional
from enum import Enum

from game import Apple
import numpy as np


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

    apple: Apple

    rays: np.ndarray[float]

    def __init__(self, board_size, piece_size):
        self.piece_size = piece_size
        self.board_size = board_size

        self.setup()

    def setup(self):
        self.pos = pg.Vector2(self.board_size // 2, self.board_size // 2)
        self.pieces = []
        self.pieces.append(self.pos.copy())
        self.move_dir = MoveDirection.RIGHT

        self.apples_eaten = 0
        self.age = 0
        self.set_new_apple()
        self.rays = np.zeros(24)

    def set_new_apple(self):
        apple = Apple(self.board_size, self.piece_size)
        apple.set_to_random_position(self.pieces)
        self.apple = apple

    def handle_apple_collision(self):
        if self.pos == self.apple.pos:
            self.grow()
            self.apples_eaten += 1
            self.apple.set_to_random_position(self.pieces)

    def update(self):
        self.pos += self.get_dir_vector(self.move_dir)
        self.pieces_update()

        self.handle_walls()
        self.handle_self_collision()
        self.handle_apple_collision()

        self.update_rays()
        self.age += 1

    def pieces_update(self):
        self.pieces.pop()
        self.pieces.insert(0, self.pos.copy())

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
            pg.draw.rect(
                screen,
                (50, 230, 0),
                pg.Rect(piece.x * self.piece_size, piece.y * self.piece_size, self.piece_size, self.piece_size),
                10
            )

        self.apple.draw(screen)

    def handle_event(self, event: pg.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.move_dir = MoveDirection.LEFT
            elif event.key == pygame.K_RIGHT:
                self.move_dir = MoveDirection.RIGHT
            elif event.key == pygame.K_DOWN:
                self.move_dir = MoveDirection.DOWN
            if event.key == pygame.K_UP:
                self.move_dir = MoveDirection.UP

    def handle_walls(self):
        if self.pos.x < 0 or self.pos.x > self.board_size - 1 or self.pos.y < 0 or self.pos.y > self.board_size - 1:
            self.restart()

    def update_rays(self):

        DIAG = 1.4
        MAX_DISTANCE = self.board_size * DIAG
        # walls

        self.rays[RaysDirections.UP.value] = self.pos.y
        self.rays[RaysDirections.RIGHT.value] = self.board_size - self.pos.x
        self.rays[RaysDirections.DOWN.value] = self.board_size - self.pos.y
        self.rays[RaysDirections.LEFT.value] = self.pos.x

        self.rays[RaysDirections.UP_RIGHT.value] = min(self.rays[RaysDirections.UP.value],
                                                       self.rays[RaysDirections.RIGHT.value]) * DIAG
        self.rays[RaysDirections.DOWN_RIGHT.value] = min(self.rays[RaysDirections.DOWN.value],
                                                         self.rays[RaysDirections.RIGHT.value]) * DIAG
        self.rays[RaysDirections.DOWN_LEFT.value] = min(self.rays[RaysDirections.DOWN.value],
                                                        self.rays[RaysDirections.LEFT.value]) * DIAG
        self.rays[RaysDirections.UP_LEFT.value] = min(self.rays[RaysDirections.UP.value],
                                                      self.rays[RaysDirections.LEFT.value]) * DIAG

        # apple

        apple_dir = self.get_direction_between_vectors(self.pos, self.apple.pos)
        diff = (self.apple.pos - self.pos)

        for direction in range(len(RaysDirections)):
            self.rays[len(RaysDirections) + direction] = MAX_DISTANCE

        if apple_dir:
            self.rays[len(RaysDirections) + apple_dir.value] = diff.magnitude()

        # self

        smallest = np.repeat(MAX_DISTANCE, len(RaysDirections))
        for piece in self.pieces[1:]:
            piece_dir = self.get_direction_between_vectors(self.pos, piece)
            if piece_dir:
                dist = (piece - self.pos).magnitude()
                if dist < smallest[piece_dir.value]:
                    smallest[piece_dir.value] = dist

        for direction in range(len(RaysDirections)):
            self.rays[len(RaysDirections) * 2 + direction] = smallest[direction]

        # normalizing

        self.rays /= MAX_DISTANCE

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
            self.restart()

    def restart(self):
        self.setup()
