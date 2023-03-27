from __future__ import annotations

import pygame as pg


class Block(pg.Rect):
    def __init__(self, pos: pg.Vector2, piece_size: int, color: pg.Color):
        self.padding = 8
        super().__init__((pos.x *piece_size) + self.padding, (pos.y*piece_size)+self.padding, piece_size-self.padding, piece_size-self.padding)
        self.color = color
        self.piece_size = piece_size

    def draw(self, screen: pg.Surface):
        pg.draw.rect(screen, self.color, self, self.piece_size-self.padding, 5)

