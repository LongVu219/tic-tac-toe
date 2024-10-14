import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

class TicTacToeEnv:
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.current_player = 1
        return self.get_state()
    
    def get_state(self):
        return str(self.board.tolist())
    
    def print_board(self):
        for i in range (0, self.board_size):
            for j in range (0, self.board_size):
                if (self.board[i, j] == 1):
                    print('x', end ="")
                elif (self.board[i, j] == -1):
                    print('o', end ="")
                else:
                    print('.')
        print("")
    
    def get_valid_moves(self):
        move = []
        for i in range (self.board_size):
            for j in range (self.board_size):
                if (self.board[i][j] == 0):
                    move.append((i,j))
        return move
    
    def apply(self, row, col):
        self.board[row, col] = self.current_player
        self.current_player *= -1

    
    
    def check_win(self):
        for i in range(self.board_size):
            if (abs(sum(self.board[i])) == self.board_size) or abs(sum(self.board[:, i])) == self.board_size:
                return True
        
        if abs(sum([self.board[i][i] for i in range(self.board_size)])) == self.board_size:
            return True
        if abs(sum([self.board[i][self.board_size-1-i] for i in range(self.board_size)])) == self.board_size:
            return True
        
        return False