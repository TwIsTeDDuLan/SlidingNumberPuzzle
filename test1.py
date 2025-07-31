import numpy as np
import random
import copy

class SlidingPuzzle:
    def __init__(self, size=3, array=[]):
        self.size = size
        self.array = array
        if array == []:
            self.current_state = self.randomPuzzle()
            self.size = len(self.array)
        else:
            self.printPuzzle()
    
    def randomPuzzle(self):
        return 1
    
    def printPuzzle(self):
        """Print the current state of the puzzle"""
        for row in self.array:
            print(" ".join(str(x) for x in row))
        print()
        
        
if __name__ == "__main__":
        puzzle = SlidingPuzzle(size=3, array=[[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        puzzle.printPuzzle()
        # Additional functionality can be added here