import sys
import pygame as pg
from time import sleep
from game import Snake, Apple, Board, MoveDirection
from nn import NeuralNetwork
from game_deepq import DeepQAgent
from game_deepq.deepq_agent import DeepQSnake
import numpy as np

pg.init()

size = WIDTH, HEIGHT = 800, 800
screen = pg.display.set_mode(size)

board_size = 20
piece_size = WIDTH // board_size

epochs = 10
epoch_size = 1000
agent = DeepQAgent(epochs, epoch_size, board_size, piece_size)

agent.train()

num_games = 10000
for game in range(num_games):
    snake = DeepQSnake(board_size, piece_size)
    snake.current_state = snake.get_state()

    FRAME_RATE = 60
    clock = pg.time.Clock()
    dt: float = 0.0
    counter: float = 0.0
    DELAY = 0.125

    while snake.is_alive():
        board = Board(board_size, piece_size)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

            snake.handle_event(event)

        if counter >= DELAY:
            action = agent.agentNetwork.predict(snake.current_state)
            action = MoveDirection(np.argmax(action))
            next_state, reward, done = snake.step(action)
            snake.current_state = next_state
            counter = 0

        screen.fill(pg.color.THECOLORS["black"])

        board.draw(screen)

        snake.draw(screen)

        pg.display.flip()
        dt = clock.tick(FRAME_RATE) / 1000.0
        counter += dt

    print(f"Game {game+1}: Score = {snake.get_score()}")

print("All games completed")
