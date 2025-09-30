import numpy as np
import random
import copy
import math
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class SlidPuzzleTraingData():
    """"here we tried to exclude redundent moves such as "Up, Down, Up, Down" we could ignoor UP in the first place resulting "Up,Down,Left"."""
    
    def __init__(self, size=3,size_of_training_data=1000):
        self.size = size
        self.goal_state = self.generate_goal_state()
        self.training_data = self.generate_training_data(size_of_training_data)
        
    def generate_goal_state(self):
        goal = np.arange(1, self.size * self.size + 1).reshape(self.size, self.size)
        goal[-1, -1] = 0
        return goal
    
    def generate_training_data(self, size_of_training_data):
        puzzle = self.goal_state.copy()
        training_data = []
        redundentMove = ""
        redundentCount = 0
        
        
        
        for _ in range(size_of_training_data):
            data = []
            valid_moves = []
            target_move, manhattan_distance, state = None, None, None
            
            empty_row, empty_col = np.where(puzzle == 0)
            empty_row, empty_col = empty_row[0], empty_col[0]
            
            #finds valid moves relative to the current possition
            if empty_row > 0: valid_moves.append('up')
            if empty_row < self.size - 1: valid_moves.append('down')  
            if empty_col > 0: valid_moves.append('left')
            if empty_col < self.size - 1: valid_moves.append('right')
            
            move = random.choice(valid_moves)
            #choses a random move
            if move != redundentMove and redundentCount < 1:
                redundentMove = move
                redundentCount+=1
            else:
                redundentCount-=1
                continue
            
            #moves the puzzle and reverse the move to make the target move
            if move == 'up':
                target_move = 'down'
                puzzle[empty_row, empty_col], puzzle[empty_row - 1, empty_col] = puzzle[empty_row - 1, empty_col], puzzle[empty_row, empty_col]
            elif move == 'down':
                target_move = 'up'
                puzzle[empty_row, empty_col], puzzle[empty_row + 1, empty_col] = puzzle[empty_row + 1, empty_col], puzzle[empty_row, empty_col]
            elif move == 'left':
                target_move = 'right'
                puzzle[empty_row, empty_col], puzzle[empty_row, empty_col - 1] = puzzle[empty_row, empty_col - 1], puzzle[empty_row, empty_col]
            elif move == 'right':
                target_move = 'left'
                puzzle[empty_row, empty_col], puzzle[empty_row, empty_col + 1] = puzzle[empty_row, empty_col + 1], puzzle[empty_row, empty_col]
            
            data.append(puzzle.copy().flatten())
            data.append(target_move)
            
            #calculates manhatten distance
            distance = 0
            for i in range(self.size):
                for j in range(self.size):
                    if puzzle[i, j] != 0:
                        goal_pos = np.where(self.goal_state == puzzle[i, j])
                        goal_i, goal_j = goal_pos[0][0], goal_pos[1][0]
                        distance += abs(i - goal_i) + abs(j - goal_j)
            manhattan_distance = distance
            data.append(manhattan_distance)
            #appends the data to the training set
            training_data.append(data)

        return training_data
    

if __name__=="__main__":
    puzle = SlidPuzzleTraingData(3,5)
    print(puzle.training_data)