import numpy as np
import random
import copy
import math

class SlidingPuzzle:
    """Class to represent a sliding puzzle game."""
    def __init__(self, size=3, array=None):
        self.size = size if isinstance(size, int) else round(size)
        self.wsize = self.size * self.size
        self.current_state = array
        if array is None:
            self.current_state = self.randomPuzzle()
        else:
            #currentyl we assume the array is a valid puzzle state
            self.array = np.array(array)
            self.size = len(self.array)
            #self.checkPuzzle()
        self.index = np.where(self.current_state == 0)
        self.index_row = self.index[0][0]
        self.index_col = self.index[1][0]
    
    def randomPuzzle(self):
        arr = list(range(self.wsize))
        random.shuffle(arr)
        return np.array(arr).reshape((self.size, self.size))
        
    
    def currentState(self):
        """Print the current state of the puzzle"""
        print("Current State of the Puzzle:\n")
        for row in self.current_state:
            print("  ".join(f"{x:3}" for x in row))
            print()
        print()
        
    def up(self):
        """Move the empty tile up if possible."""
        if self.index_row > 0:
            new_state = copy.deepcopy(self.current_state)
            new_state[self.index_row][self.index_col], new_state[self.index_row - 1][self.index_col] = \
                new_state[self.index_row - 1][self.index_col], new_state[self.index_row][self.index_col]
            self.current_state = new_state
            return None
        else:
            print("Cannot move up, already at the top row.")
            return None
    
    def down(self):
        if self.index_row < self.size - 1:
            new_state = copy.deepcopy(self.current_state)
            new_state[self.index_row][self.index_col], new_state[self.index_row + 1][self.index_col] = \
                new_state[self.index_row + 1][self.index_col], new_state[self.index_row][self.index_col]
            self.current_state = new_state
            return None
        else:
            print("Cannot move down, already at the bottom row.")
            return None
    
    def left(self):
        if self.index_col > 0:
            new_state = copy.deepcopy(self.current_state)
            new_state[self.index_row][self.index_col], new_state[self.index_row][self.index_col - 1] = \
                new_state[self.index_row][self.index_col - 1], new_state[self.index_row][self.index_col]
            self.current_state = new_state
            return None
        else:
            print("Cannot move left, already at the leftmost column.")
            return None
    
    def right(self):
        if self.index_col < self.size - 1:
            new_state = copy.deepcopy(self.current_state)
            new_state[self.index_row][self.index_col], new_state[self.index_row][self.index_col + 1] = \
                new_state[self.index_row][self.index_col + 1], new_state[self.index_row][self.index_col]
            self.current_state = new_state
            return None
        else:
            print("Cannot move right, already at the rightmost column.")
            return None
        
        
if __name__ == "__main__":
        #puzzle = SlidingPuzzle(size=3, array=[[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        puzzle = SlidingPuzzle(size=3)
        puzzle.currentState()
        puzzle.up()
        puzzle.currentState()