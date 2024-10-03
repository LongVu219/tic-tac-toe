import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

class ValueIteration:
    def __init__(self, env, gamma = 0.99):
        self.env = env
        self.gamma = gamma
        self.values = defaultdict(float)
        self.policy = {}
    
    def train(self, num_iterations = 100):
        for _ in range(num_iterations):
            delta = 0
            states_to_update = set()
            
            self.env.reset()
            self.generate_states(self.env.get_state(), states_to_update)
            
            for state in states_to_update:
                board = eval(state)
                self.env.board = np.array(board)
                tmp = self.values[state]
                self.values[state] = self.get_max_value(state)
                delta = max(delta, abs(tmp - self.values[state]))
    
    def generate_states(self, state, states):
        states.add(state)
        board = eval(state)
        self.env.board = np.array(board)
        
        if self.env.check_win() or len(self.env.get_valid_moves()) == 0:
            return
        
        for move in self.env.get_valid_moves():
            self.env.board = np.array(board)
            next_state, m1, m2 = self.env.make_move(move[0], move[1])
            if next_state not in states:
                self.generate_states(next_state, states)
    
    def get_max_value(self, state):
        board = eval(state)
        self.env.board = np.array(board)
        
        if self.env.check_win():
            if (self.env.current_player == 1): 
                return 1
            return -1
        
        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            return 0
        
        values = []
        for move in valid_moves:
            self.env.board = np.array(board)
            nxt_state, value, done = self.env.make_move(move[0], move[1])
            if done:
                values.append(value)
            else:
                values.append(value + self.gamma * -self.get_max_value(nxt_state))
        
        return max(values)
    
    def get_action(self, state):
        board = eval(state)
        self.env.board = np.array(board)
        valid_moves = self.env.get_valid_moves()
        
        if not valid_moves:
            return None
        
        best_value = float('-inf')
        best_move = None
        
        for move in valid_moves:
            self.env.board = np.array(board)
            next_state, reward, done = self.env.make_move(move[0], move[1])
            if done:
                value = reward
            else:
                value = reward + self.gamma * -self.values[next_state]
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move