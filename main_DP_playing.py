import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from ValueIteration import ValueIteration
from TicTacToeEnv import TicTacToeEnv

def play_game(env, agent):
    state = env.reset()
    done = False
    
    while not done:
        if env.current_player == 1:
            action = agent.get_action(state)
        else:
            valid_moves = env.get_valid_moves()
            action = random.choice(valid_moves) if valid_moves else None
        
        if action is None:
            break
        
        state, reward, done = env.make_move(action[0], action[1])
    
    if env.check_win():
        return 1 if env.current_player == -1 else -1
    return 0

def evaluate_agent(env, agent, num_games=100):
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(num_games):
        result = play_game(env, agent)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    
    return wins / num_games, losses / num_games, draws / num_games

board_sizes = [4, 5]
results = {}

for size in board_sizes:

    print(f'Board size : {size}x{size}' + '-' * 30)

    env = TicTacToeEnv(board_size=size)
    agent = ValueIteration(env)
    agent.train(num_iterations=100)
    
    win_rates = []
    eval_intervals = list(range(0, 1001, 100))
    
    for i in eval_intervals:
        if i > 0:
            agent.train(num_iterations=100)
        win_rate, loss_rate, draw_rate = evaluate_agent(env, agent)
        win_rates.append(win_rate)
        print(f'Epoch number {i} win rate : {win_rate}')
    
    results[size] = (eval_intervals, win_rates)