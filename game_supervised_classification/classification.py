import sys
import pygame as pg
from time import sleep
from game import Snake, Apple, Board
from game_supervised_classification.classification_snake import SupervisedClassificationSnake
from game_supervised_classification.supervised_cassification_Network import SupervisedNeuralNetwork

pg.init()

size = WIDTH, HEIGHT = 800, 800
screen = pg.display.set_mode(size)

board_size = 20
piece_size = WIDTH // board_size

model = SupervisedNeuralNetwork()
snake = SupervisedClassificationSnake(board_size, piece_size, model)

FRAME_RATE = 60
clock = pg.time.Clock()
dt: float = 0.0
counter: float = 0.0
DELAY = 0.125

board = Board(board_size, piece_size)

def tick():
    snake.update(verbose=1)


while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()


    # updates
    if counter >= DELAY:
        tick()

        counter = 0
    if not snake.alive:
        snake = SupervisedClassificationSnake(board_size, piece_size, model)
    # drawing
    screen.fill(pg.color.THECOLORS["black"])

    board.draw(screen)

    snake.draw(screen)

    pg.display.flip()
    dt = clock.tick(FRAME_RATE) / 1000.0
    counter += dt
