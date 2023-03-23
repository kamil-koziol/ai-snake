from __future__ import annotations

from typing import List

import pygame as pg
from random import randint


class Apple:
    pos: pg.Vector2
    board_size: int
    piece_size: int

    def __init__(self, board_size: int, piece_size: int):
        self.board_size = board_size
        self.piece_size = piece_size

    def find_available_place(self, pieces: List[pg.Vector2]):
        while True:

            rnd_vec = pg.Vector2(
                randint(0, self.board_size - 1),
                randint(0, self.board_size - 1)
            )

            if rnd_vec not in pieces:
                return rnd_vec

    def set_to_random_position(self, taken_positions: List[pg.Vector2]):
        self.pos = self.find_available_place(taken_positions)

    def draw(self, screen: pg.Surface):
        pg.draw.rect(screen,
                     (255, 50, 0),
                     pg.Rect(self.pos.x * self.piece_size, self.pos.y * self.piece_size, self.piece_size,
                             self.piece_size)
                     )
