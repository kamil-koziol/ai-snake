import sys
import pygame as pg
from time import sleep
from game import Snake, Apple, Board, MoveDirection
from nn import NeuralNetwork
from game_deepq import DeepQAgent
from game_deepq.deepq_agent import DeepQSnake, DeepQAgentConfig
import numpy as np
import torch

import matplotlib.pyplot as plt


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
    print(f"CUDA: {torch.cuda.is_available()}")

    size = WIDTH, HEIGHT = 800, 800
    screen = pg.display.set_mode(size)
    FRAME_RATE = 60
    DELAY = 0.01
    pg.init()

    board_size = 7
    piece_size = WIDTH // board_size

    config = DeepQAgentConfig(
        board_size=board_size,
        piece_size=piece_size,

        epochs=1000,
        epoch_steps=1024 * 40 / 1024,

        game_steps_per_epoch=1024 * 4,
        memory_size=1024 * 100,


        epsilon_decay=0.999,
        epsilon_min=0.01,
        gamma=0.9,

        lr=0.0005,
        batch_size=1024,

        dataloader_shuffle=True,
        dataloader_num_workers=0,

        hidden_sizes=[512],
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    agent = DeepQAgent(config)

    losses = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, 1)
    ax.set_yscale('log')
    line, = ax.plot(losses)



    def after_epoch_callback(epoch_num, loss, epsilon):
        # draw loss on plot
        print(f"Epoch {epoch_num}\n\tLoss = {loss}\n\tEpsilon = {epsilon}")
        losses.append(loss)

        line.set_ydata(losses)
        line.set_xdata(range(epoch_num+1))
        ax.relim()
        ax.set_xlim(0, epoch_num)
        ax.autoscale_view(True,True,True)
        fig.canvas.draw()
        plt.pause(0.1)

        if epoch_num % 10 == 0:
            play_game(1, epoch_num, screen, board_size, piece_size)

    agent.train(after_epoch_callback=after_epoch_callback)
