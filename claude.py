import numpy as np
import random
import copy

class SlidingPuzzle:
    """Improved sliding puzzle class optimized for ML training."""
    
    def __init__(self, size=3, array=None):
        self.size = size if isinstance(size, int) else round(size)
        self.wsize = self.size * self.size
        
        if array is None:
            self.current_state = self._create_solvable_puzzle()
        else:
            self.current_state = np.array(array).reshape(self.size, self.size)
            if not self.is_solvable():
                print("Warning: Provided puzzle may not be solvable!")
        
        self._update_empty_position()
        self.goal_state = self._create_goal_state()
        self.move_count = 0
    
    def _create_goal_state(self):
        """Create the solved state."""
        goal = list(range(1, self.wsize)) + [0]
        return np.array(goal).reshape(self.size, self.size)
    
    def _create_solvable_puzzle(self):
        """Create a solvable puzzle by shuffling from goal state."""
        state = self._create_goal_state()
        # Make random valid moves to ensure solvability
        for _ in range(1000):
            valid_moves = self.get_valid_moves(state)
            if valid_moves:
                move = random.choice(valid_moves)
                state = self._apply_move(state, move)
        return state
    
    def _update_empty_position(self):
        """Update the position of the empty tile (0)."""
        self.index = np.where(self.current_state == 0)
        self.index_row = self.index[0][0]
        self.index_col = self.index[1][0]
    
    def get_valid_moves(self, state=None):
        """Get list of valid move directions."""
        if state is None:
            state = self.current_state
        
        # Find empty position for the given state
        empty_pos = np.where(state == 0)
        row, col = empty_pos[0][0], empty_pos[1][0]
        
        valid_moves = []
        if row > 0: valid_moves.append('up')
        if row < self.size - 1: valid_moves.append('down')
        if col > 0: valid_moves.append('left')
        if col < self.size - 1: valid_moves.append('right')
        
        return valid_moves
    
    def _apply_move(self, state, direction):
        """Apply a move to a state and return new state (immutable)."""
        new_state = copy.deepcopy(state)
        empty_pos = np.where(state == 0)
        row, col = empty_pos[0][0], empty_pos[1][0]
        
        if direction == 'up' and row > 0:
            new_state[row, col], new_state[row-1, col] = new_state[row-1, col], new_state[row, col]
        elif direction == 'down' and row < self.size - 1:
            new_state[row, col], new_state[row+1, col] = new_state[row+1, col], new_state[row, col]
        elif direction == 'left' and col > 0:
            new_state[row, col], new_state[row, col-1] = new_state[row, col-1], new_state[row, col]
        elif direction == 'right' and col < self.size - 1:
            new_state[row, col], new_state[row, col+1] = new_state[row, col+1], new_state[row, col]
        
        return new_state
    
    def move(self, direction):
        """Make a move and return success status."""
        if direction not in self.get_valid_moves():
            return False
        
        self.current_state = self._apply_move(self.current_state, direction)
        self._update_empty_position()
        self.move_count += 1
        return True
    
    # Keep your original methods for backwards compatibility
    def up(self):
        return self.move('up')
    
    def down(self):
        return self.move('down')
    
    def left(self):
        return self.move('left')
    
    def right(self):
        return self.move('right')
    
    def is_solved(self):
        """Check if puzzle is in solved state."""
        return np.array_equal(self.current_state, self.goal_state)
    
    def is_solvable(self):
        """Check if puzzle is solvable using inversion count."""
        flat = self.current_state.flatten()
        
        # Remove the empty tile (0) for inversion counting
        numbers = [x for x in flat if x != 0]
        
        # Count inversions
        inversions = 0
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if numbers[i] > numbers[j]:
                    inversions += 1
        
        # For odd grid width, puzzle is solvable if inversions are even
        if self.size % 2 == 1:
            return inversions % 2 == 0
        
        # For even grid width, consider empty tile position
        empty_row = np.where(self.current_state == 0)[0][0]
        empty_row_from_bottom = self.size - empty_row
        
        if empty_row_from_bottom % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0
    
    def manhattan_distance(self):
        """Calculate Manhattan distance heuristic."""
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.current_state[i, j] != 0:
                    # Find goal position
                    goal_pos = np.where(self.goal_state == self.current_state[i, j])
                    goal_i, goal_j = goal_pos[0][0], goal_pos[1][0]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance
    
    def get_state_vector(self):
        """Get flattened state as vector for ML models."""
        return self.current_state.flatten()
    
    def get_state_hash(self):
        """Get hashable representation of state."""
        return tuple(self.current_state.flatten())
    
    def reset(self):
        """Reset to a new random solvable puzzle."""
        self.current_state = self._create_solvable_puzzle()
        self._update_empty_position()
        self.move_count = 0
    
    def set_state(self, state):
        """Set puzzle to specific state."""
        self.current_state = np.array(state).reshape(self.size, self.size)
        self._update_empty_position()
        self.move_count = 0
    
    def copy(self):
        """Create a copy of the puzzle."""
        new_puzzle = SlidingPuzzle(self.size)
        new_puzzle.current_state = copy.deepcopy(self.current_state)
        new_puzzle._update_empty_position()
        new_puzzle.move_count = self.move_count
        return new_puzzle
    
    def currentState(self):
        """Print the current state of the puzzle."""
        print(f"Current State (Moves: {self.move_count}, Manhattan: {self.manhattan_distance()}):")
        for row in self.current_state:
            print("                    ", end="")
            print("  ".join(f"{x:3}" if x != 0 else "   " for x in row))
        print()
    
    def step(self, action):
        """Gym-style step function for RL training."""
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        actions = ['up', 'down', 'left', 'right']
        
        if action < 0 or action >= len(actions):
            return self.get_state_vector(), -10, False, {'invalid_move': True}
        
        direction = actions[action]
        old_manhattan = self.manhattan_distance()
        
        success = self.move(direction)
        
        if not success:
            # Invalid move penalty
            return self.get_state_vector(), -10, False, {'invalid_move': True}
        
        new_manhattan = self.manhattan_distance()
        
        # Reward calculation
        if self.is_solved():
            reward = 100  # Big reward for solving
        else:
            # Small reward for getting closer, penalty for moving away
            reward = (old_manhattan - new_manhattan) * 1.0 - 0.1  # Small step penalty
        
        done = self.is_solved()
        info = {
            'manhattan_distance': new_manhattan,
            'move_count': self.move_count,
            'invalid_move': False
        }
        
        return self.get_state_vector(), reward, done, info

# Example usage
if __name__ == "__main__":
    # Test the improved puzzle
    puzzle = SlidingPuzzle(3)
    
    print("=== Testing Improved Sliding Puzzle ===")
    puzzle.currentState()
    
    print(f"Is solvable: {puzzle.is_solvable()}")
    print(f"Is solved: {puzzle.is_solved()}")
    print(f"Valid moves: {puzzle.get_valid_moves()}")
    print(f"Manhattan distance: {puzzle.manhattan_distance()}")
    
    # Test moves
    print("\nMaking some moves:")
    for direction in ['up', 'right', 'down']:
        success = puzzle.move(direction)
        print(f"Move {direction}: {'Success' if success else 'Failed'}")
    
    puzzle.currentState()
    
    # Test RL-style step function
    print("\nTesting RL step function:")
    state, reward, done, info = puzzle.step(0)  # Try moving up
    print(f"Reward: {reward}, Done: {done}, Info: {info}")