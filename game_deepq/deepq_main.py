import sys
import pygame as pg
from time import sleep
from game import Snake, Apple, Board, MoveDirection
from nn import NeuralNetwork
from game_deepq import DeepQAgent
from game_deepq.deepq_agent import DeepQSnake
import numpy as np
import torch


def play_game(num_tries, game_num, screen, board_size, piece_size):
    board = Board(board_size, piece_size)
    clock = pg.time.Clock()
    counter: float = 0.0
    dt: float = 0.0
    game = DeepQSnake(board_size, piece_size)

    print (f"Game {game_num + 1}:")
    for try_num in range(num_tries):
        while True:

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()

            # updates
            if counter >= DELAY:
                state = game.get_state()
                q_values = agent.predict(state)
                best_action = torch.argmax(q_values).item()
                game.step(best_action)

                counter = 0

            # drawing
            screen.fill(pg.color.THECOLORS["black"])

            board.draw(screen)
            game.draw(screen)

            pg.display.flip()
            dt = clock.tick(FRAME_RATE) / 1000.0
            counter += dt

            if not game.is_alive():
                print(f"\tTry {try_num + 1}: Score = {game.get_score()}")
                game.restart()
                break


if __name__ == "__main__":
    size = WIDTH, HEIGHT = 800, 800
    screen = pg.display.set_mode(size)
    FRAME_RATE = 60
    DELAY = 0.01
    pg.init()

    board_size = 20
    piece_size = WIDTH // board_size

    epochs = 10000
    epoch_size = 1000

    agent = DeepQAgent(epochs, epoch_size, board_size, piece_size, lr=0.01)


    def after_epoch_callback(epoch_num):
        play_game(1, epoch_num, screen, board_size, piece_size)


    agent.train(after_epoch_callback=after_epoch_callback)
