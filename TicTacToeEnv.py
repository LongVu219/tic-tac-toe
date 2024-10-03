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
    
    def is_valid_move(self, row, col):

        if (0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == 0):
            return True
        
        return False
    
    def get_valid_moves(self):
        move = []
        for i in range (self.board_size):
            for j in range (self.board_size):
                if (self.board[i][j] == 0):
                    move.append((i,j))
        return move
    
    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return None, -100, True 
        
        self.board[row][col] = self.current_player
        
        if self.check_win():
            return self.get_state(), 1, True
        
        if len(self.get_valid_moves()) == 0:
            return self.get_state(), 0, True
        
        self.current_player *= -1
        return self.get_state(), 0, False
    
    def check_win(self):
        for i in range(self.board_size):
            if (abs(sum(self.board[i])) == self.board_size) or abs(sum(self.board[:, i])) == self.board_size:
                return True
        
        if abs(sum([self.board[i][i] for i in range(self.board_size)])) == self.board_size:
            return True
        if abs(sum([self.board[i][self.board_size-1-i] for i in range(self.board_size)])) == self.board_size:
            return True
        
        return False