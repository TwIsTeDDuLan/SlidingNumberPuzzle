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
            self.size = len(self.array)
            self.checkPuzzle()
    
    def randomPuzzle(self):
        arr = list(range(self.wsize))
        random.shuffle(arr)
        return np.array(arr).reshape((self.size, self.size))
        
    
    def currentState(self):
        """Print the current state of the puzzle"""
        for row in self.current_state:
            print("  ".join(f"{x:3}" for x in row))
            print()
        print()
        
        
if __name__ == "__main__":
        #puzzle = SlidingPuzzle(size=3, array=[[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        puzzle = SlidingPuzzle(size=3)
        puzzle.currentState()