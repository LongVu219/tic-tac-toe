import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import copy

class TicTacToeEnv:
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()
        self.Q = {}
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.current_player = 1
        return self.get_state()
    
    def get_state(self):
        return str(self.board.reshape(self.board_size * self.board_size))
    
    def print_board(self):
        for i in range (0, self.board_size):
            for j in range (0, self.board_size):
                if (self.board[i, j] == 1):
                    print('x', end = "")
                elif (self.board[i, j] == -1):
                    print('o', end = "")
                else:
                    print('.', end = "")
            print("")
        
        print('-' * 30)
    
    def valid_moves(self):
        move = []
        for i in range (self.board_size):
            for j in range (self.board_size):
                if (self.board[i][j] == 0):
                    move.append((i,j))
        return move
    
    def apply(self, action):
        self.board[action[0], action[1]] = self.current_player
        self.current_player *= -1    
    
    def check_end(self):

        if (len(self.valid_moves()) == 0 or self.check_win() == True):
            return True

        return False

    def check_win(self):
        for i in range(self.board_size):
            if (abs(sum(self.board[i, :])) == self.board_size) or abs(sum(self.board[:, i])) == self.board_size:
                return True
        
        if abs(sum([self.board[i][i] for i in range(self.board_size)])) == self.board_size:
            return True
        if abs(sum([self.board[i][self.board_size-1-i] for i in range(self.board_size)])) == self.board_size:
            return True
        
        return False

    def state_reward(self):
        if (self.check_win() == True):
            return self.current_player * 100
        
        return -15


class Computer_PLayer():
    def __init__(self, policy = 'optimal', epsilon = None, decay_rate = None):
        self.policy = policy
        self.Q = {}
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def Q_value(self, state, action):
        if (self.Q.get((state, action)) is None):
            return 0
        
        else: 
            return self.Q.get((state, action))

    def choose_action(self, env : TicTacToeEnv, episode = None):
        possible_action = env.valid_moves()

        if (self.policy == 'random'):
            return random.choice(possible_action)
        
        def choose_optimal():
            if (env.current_player == 1):
                best_val = -99999
                best_action = possible_action[0]
                for action in possible_action:
                    new_env = copy.deepcopy(env)
                    new_env.apply(action)
                    val = self.Q_value(env.get_state(), action)
                    if (best_val < val):
                        best_val = val
                        best_action = action
                return best_action
            
            else:
                best_val = 99999
                best_action = possible_action[0]
                for action in possible_action:
                    new_env = copy.deepcopy(env)
                    new_env.apply(action)
                    val = self.Q_value(env.get_state(), action)
                    if (best_val > val):
                        best_val = val
                        best_action = action
                return best_action

        if (self.policy == 'optimal'):
            return choose_optimal()

        elif (self.policy == 'epsilon'):
            rd = np.random.uniform(0, 1)
            eps = self.epsilon * (self.decay_rate**(episode//6000))
            eps = max(eps, 0.1)
            if (rd <= eps):
                return random.choice(possible_action)
            else:
                return choose_optimal()


cpu1 = Computer_PLayer(policy = "epsilon", epsilon = 0.85, decay_rate = 0.97)    
cpu2 = Computer_PLayer(policy = "random")       
episode = 720000 * 2
alpha = 0.9
gamma = 0.9


#tmp = board = TicTacToeEnv()
#board.reset()
#action = (1, 1)
#board.apply(action)
#board.print_board()

cpu1_win = 0
cpu2_win = 0
draw = 0
batch_size = 800 * 5

for i in range (episode):
    board = TicTacToeEnv(board_size=4)
    board.reset()
    
    current_cpu = cpu1
    state = board.get_state()
    action = current_cpu.choose_action(board, episode = 0)

    while(board.check_end() == False):
        #board.print_board()
        #print(board.valid_moves())
        cur_episode = None
        if (board.current_player == 1):
            current_cpu = cpu1
            cur_episode = i
        else:
            current_cpu = cpu2

        board.apply(action)
        new_state = board.get_state()

        if (len(board.valid_moves()) > 0):
            new_action = current_cpu.choose_action(board, episode = cur_episode)
        

        R = board.state_reward()

        cpu1.Q[(state, action)] = cpu1.Q_value(state, action) + alpha * (R + gamma*cpu1.Q_value(new_state, new_action) - cpu1.Q_value(state, action)) 

        #print(action, '-->', new_action)
        state = new_state
        action = new_action
    
    #board.print_board()
    #print(board.current_player)

    if (board.check_win() == True):
        if (board.current_player == 1):
            cpu2_win += 1
        else:
            cpu1_win += 1
    else:
        draw += 1
    
    if (i % batch_size == 0):
        print(f'Episode number {i + 1} | CPU1 winrate : {cpu1_win/batch_size} | Draw : {draw/batch_size}')
        cpu1_win = 0
        cpu2_win = 0
        draw = 0

import pickle

with open('model/SARSA_4x4.pkl', 'wb') as f:
    pickle.dump(cpu1.Q, f)

#with open('model/SARSA.pkl', 'rb') as f:
#    cpu1.Q = pickle.load(f)

#print(cpu1.Q)
