import numpy as np
import random
from collections import deque
import copy

class SlidingPuzzle:
    def __init__(self, size=3):
        """
        Initialize a sliding puzzle
        size: dimension of the puzzle (size x size)
        0 represents the empty cell
        """
        self.size = size
        self.goal_state = self.create_goal_state()
        self.current_state = self.create_shuffled_puzzle()
    
    def create_goal_state(self):
        """Create the solved state: numbers 1 to size²-1, then 0 (empty)"""
        goal = np.arange(1, self.size * self.size).tolist() + [0]
        return np.array(goal).reshape(self.size, self.size)
    
    def create_shuffled_puzzle(self):
        """Create a solvable shuffled puzzle"""
        # Start with goal state and make random valid moves
        state = copy.deepcopy(self.goal_state)
        for _ in range(1000):  # Make 1000 random moves
            moves = self.get_valid_moves(state)
            if moves:
                move = random.choice(moves)
                state = self.make_move(state, move)
        return state
    
    def find_empty_cell(self, state):
        """Find position of empty cell (0)"""
        pos = np.where(state == 0)
        return pos[0][0], pos[1][0]
    
    def get_valid_moves(self, state):
        """Get all valid moves (directions empty cell can move)"""
        empty_row, empty_col = self.find_empty_cell(state)
        moves = []
        
        # Check each direction
        directions = [
            ('up', -1, 0),
            ('down', 1, 0),
            ('left', 0, -1),
            ('right', 0, 1)
        ]
        
        for direction, dr, dc in directions:
            new_row, new_col = empty_row + dr, empty_col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                moves.append((direction, new_row, new_col))
        
        return moves
    
    def make_move(self, state, move):
        """Make a move and return new state"""
        new_state = copy.deepcopy(state)
        empty_row, empty_col = self.find_empty_cell(state)
        direction, target_row, target_col = move
        
        # Swap empty cell with target cell
        new_state[empty_row][empty_col] = state[target_row][target_col]
        new_state[target_row][target_col] = 0
        
        return new_state
    
    def is_solved(self, state):
        """Check if puzzle is solved"""
        return np.array_equal(state, self.goal_state)
    
    def state_to_string(self, state):
        """Convert state to string for hashing/comparison"""
        return str(state.flatten())
    
    def print_state(self, state):
        """Pretty print the puzzle state"""
        for row in state:
            print(' '.join([f'{num:2d}' if num != 0 else '  ' for num in row]))
        print()
    
    def manhattan_distance(self, state):
        """Calculate Manhattan distance heuristic"""
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] != 0:
                    # Find where this number should be in goal state
                    goal_pos = np.where(self.goal_state == state[i][j])
                    goal_i, goal_j = goal_pos[0][0], goal_pos[1][0]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

# Basic BFS Solver (for comparison with ML approaches)
class BFSSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle
    
    def solve(self, max_moves=10000):
        """Solve using BFS - good baseline to compare ML against"""
        start_state = self.puzzle.current_state
        if self.puzzle.is_solved(start_state):
            return []
        
        queue = deque([(start_state, [])])
        visited = {self.puzzle.state_to_string(start_state)}
        moves_explored = 0
        
        while queue and moves_explored < max_moves:
            current_state, path = queue.popleft()
            moves_explored += 1
            
            for move in self.puzzle.get_valid_moves(current_state):
                new_state = self.puzzle.make_move(current_state, move)
                state_str = self.puzzle.state_to_string(new_state)
                
                if state_str not in visited:
                    visited.add(state_str)
                    new_path = path + [move]
                    
                    if self.puzzle.is_solved(new_state):
                        return new_path
                    
                    queue.append((new_state, new_path))
        
        return None  # No solution found within move limit

# Example usage and testing
if __name__ == "__main__":
    # Create a 3x3 puzzle
    puzzle = SlidingPuzzle(size=3)
    
    print("Initial puzzle state:")
    puzzle.print_state(puzzle.current_state)
    
    print("Goal state:")
    puzzle.print_state(puzzle.goal_state)
    
    print(f"Manhattan distance: {puzzle.manhattan_distance(puzzle.current_state)}")
    print(f"Is solved: {puzzle.is_solved(puzzle.current_state)}")
    
    # Test BFS solver
    print("\nSolving with BFS...")
    solver = BFSSolver(puzzle)
    solution = solver.solve(max_moves=5000)
    
    if solution:
        print(f"Solution found in {len(solution)} moves!")
        print("First few moves:", solution[:5])
    else:
        print("No solution found within move limit")
    
    # Test valid moves
    print(f"\nValid moves from current state:")
    for i, move in enumerate(puzzle.get_valid_moves(puzzle.current_state)):
        print(f"{i+1}. Move {move[0]}")