from __future__ import annotations
import pygame as pg
from game import Block

class Board:
    def __init__(self, board_size: int, piece_size: int):
        self.board_size = board_size
        self.piece_size = piece_size

    def draw(self, screen: pg.Surface):
        for row in range(self.board_size):
            for col in range(self.board_size):
                b = Block(pg.Vector2(row, col), self.piece_size, pg.Color(24, 24, 24))
                b.draw(screen)

