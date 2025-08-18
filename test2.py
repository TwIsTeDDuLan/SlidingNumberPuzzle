# Sliding Puzzle implimentation
import numpy as np
import random
import copy
<<<<<<< Updated upstream:test2.py
=======
import math
from sklearn.preprocessing import _encoders
<<<<<<< Updated upstream:test2.py
>>>>>>> Stashed changes:test1.py
=======
>>>>>>> Stashed changes:test1.py

class SlidingPuzzle:
    """Class to represent a sliding puzzle game."""
    def __init__(self, size=3):
        self.size = size if isinstance(size, int) else round(size)
        self.wsize = self.size * self.size
        self.goal = self.create_goal_state()
        self.current_state, self.solution_moves = self.create_puzzle_with_solution()
        self.index = np.where(self.current_state == 0)
        self.index_row = self.index[0][0]
        self.index_col = self.index[1][0]
        
        
    def create_goal_state(self):
        goal = list(range(1, self.wsize)) + [0]
        return np.array(goal).reshape(self.size, self.size)
        
    def is_solved(self):
        return np.array_equal(self.current_state, self.goal_state)

    def goalState(self):
        """Print the goal state of the puzzle"""
        print("Goal State of the Puzzle:")
        
        for row in self.goal:
            print("                    ",end="")
            print("  ".join(f"{x:3}" for x in row))
            print()
            
        return None
    
    def create_puzzle_with_solution(self):
        """Create a puzzle by making random moves from goal state, store the moves"""
        # Start with goal state
        puzzle = self.goal.copy()
        moves_made = []
        
        # Track empty position
        empty_row, empty_col = np.where(puzzle == 0)
        empty_row, empty_col = empty_row[0], empty_col[0]
        
        # Make random moves
        num_moves = random.randint(20, 100)  # Random difficulty
        
        for _ in range(num_moves):
            # Get valid moves
            valid_moves = []
            if empty_row > 0: valid_moves.append('up')
            if empty_row < self.size - 1: valid_moves.append('down')  
            if empty_col > 0: valid_moves.append('left')
            if empty_col < self.size - 1: valid_moves.append('right')
            
            # Pick random move
            move = random.choice(valid_moves)
            moves_made.append(move)
            
            # Apply move and update empty position
            if move == 'up':
                puzzle[empty_row, empty_col], puzzle[empty_row-1, empty_col] = puzzle[empty_row-1, empty_col], puzzle[empty_row, empty_col]
                empty_row -= 1
            elif move == 'down':
                puzzle[empty_row, empty_col], puzzle[empty_row+1, empty_col] = puzzle[empty_row+1, empty_col], puzzle[empty_row, empty_col]  
                empty_row += 1
            elif move == 'left':
                puzzle[empty_row, empty_col], puzzle[empty_row, empty_col-1] = puzzle[empty_row, empty_col-1], puzzle[empty_row, empty_col]
                empty_col -= 1
            elif move == 'right':
                puzzle[empty_row, empty_col], puzzle[empty_row, empty_col+1] = puzzle[empty_row, empty_col+1], puzzle[empty_row, empty_col]
                empty_col += 1
        
        # Reverse moves for solution
        solution = []
        for move in reversed(moves_made):
            if move == 'up': solution.append('down')
            elif move == 'down': solution.append('up')  
            elif move == 'left': solution.append('right')
            elif move == 'right': solution.append('left')
        
        return puzzle, solution 
        
    def currentState(self):
        """Print the current state of the puzzle"""
        print("Current State of the Puzzle:")
        
        for row in self.current_state:
            print("                    ",end="")
            print("  ".join(f"{x:3}" for x in row))
            
    def manhattan_distance(self):
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.current_state[i,j] != 0:  # Skip empty tile
                    # Find where this number should be
                    goal_pos = np.where(self.goal == self.current_state[i,j])
                    goal_i, goal_j = goal_pos[0][0], goal_pos[1][0]
                    # Add distance
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance
        
    def up(self):
        """Move the empty tile up if possible."""
        if self.index_row > 0:
            new_state = copy.deepcopy(self.current_state)
            new_state[self.index_row][self.index_col], new_state[self.index_row - 1][self.index_col] = \
                new_state[self.index_row - 1][self.index_col], new_state[self.index_row][self.index_col]
            self.index_row -= 1
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
            self.index_row += 1
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
            self.index_col -= 1
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
            self.index_col += 1
            return None
        else:
            print("Cannot move right, already at the rightmost column.")
            return None
        
        
if __name__ == "__main__":
    puzzle = SlidingPuzzle(3)
    puzzle.goalState()
    puzzle.currentState()
    
    print("Manhattan Distance:", puzzle.manhattan_distance())