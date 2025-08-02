import numpy as np
import random
import copy
import math
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Sliding Puzzle implimentation
class SlidingPuzzle:
    """Class to represent a sliding puzzle game."""
    def __init__(self, size=3):
        self.size = size if isinstance(size, int) else round(size)
        self.wsize = self.size * self.size
        self.goal = self.create_goal_state()
        self.current_state, self.solution_moves, self.distance_log = self.create_puzzle_with_solution()
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
        distance_log = []
        
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
                
            # Calculate Manhattan distance for the final state
            distance = self.manhattan_distance(puzzle)
            distance_log.append(distance)
        
        # Reverse moves for solution
        solution = []
        for move in reversed(moves_made):
            if move == 'up': solution.append('down')
            elif move == 'down': solution.append('up')  
            elif move == 'left': solution.append('right')
            elif move == 'right': solution.append('left')
        
        # Reverse the manhattan distances
        distance_log = distance_log[::-1]
        
        return puzzle, solution, distance_log 
        
    def currentState(self):
        """Print the current state of the puzzle"""
        print("Current State of the Puzzle:")
        
        for row in self.current_state:
            print("                    ",end="")
            print("  ".join(f"{x:3}" for x in row))
            
    def manhattan_distance(self, state):
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if state[i, j] != 0:
                    goal_pos = np.where(self.goal == state[i, j])
                    goal_i, goal_j = goal_pos[0][0], goal_pos[1][0]
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
        
# Generate test data
def generate_training_data(num_samples=100, size=3):
    print(f"Generating {num_samples} training samples...")
    
    puzzles = []
    solutions = []
    manhattan_distances = []
    
    for i in range(num_samples):
        if i % 1000 == 0:
            print(f"Generated {i}/{num_samples} samples...")
        
        # Create a single puzzle generator (reuse it)
        puzzle_gen = SlidingPuzzle(size)
        
        # Get the data from the puzzle that was already created
        current_state = puzzle_gen.current_state
        solution = puzzle_gen.solution_moves
        
        puzzles.append(current_state.flatten())
        manhattan_distances.append(puzzle_gen.manhattan_distance(current_state))
        
        if len(solution) > 0:
            solutions.append(solution[0])
        else:
            solutions.append('no_move')
    
    return np.array(puzzles), np.array(solutions), np.array(manhattan_distances)
    
def prepare_training_data(x, y, z):
    """Prepare training data for the sliding puzzle."""
    
    # Convert to numpy arrays
    x = np.array(x)  # puzzles (shape: num_samples, 9)
    y = np.array(y)  # solutions  
    z = np.array(z)  # manhattan distances (shape: num_samples,)

    # COMBINE puzzles and manhattan distances
    # Add manhattan distance as an extra column to each puzzle
    z_reshaped = z.reshape(-1, 1)  # Make it (num_samples, 1)
    x_combined = np.hstack([x, z_reshaped])  # Combine: (num_samples, 10)

    # Convert move strings to integers
    move_encoder = LabelEncoder()
    move_encoder.fit(['up', 'down', 'left', 'right', 'no_move'])
    y_encoded = move_encoder.fit_transform(y)

    print("Moves Encoding:")
    for i, move in enumerate(['up', 'down', 'left', 'right', 'no_move']):
        print(f"    {move} -> {i}")

    print(f"\nData ready for training:")
    print(f"X_combined shape: {x_combined.shape}")  # Should be (num_samples, 10)
    print(f"Y_encoded shape: {y_encoded.shape}")    # Should be (num_samples,)

    return x_combined, y_encoded, move_encoder  # Return combined data
    
def create_puzzle_model():
    model = models.Sequential([
        # OLD: input_shape=(9,)
        # NEW: input_shape=(10,)  ← One extra input for Manhattan distance
        layers.Dense(128, activation='relu', input_shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_model_prediction(model, encoder):
    """Test the model on a new puzzle"""

    # Create a test puzzle
    test_puzzle = SlidingPuzzle(3)
    puzzle_state, true_solution = test_puzzle.create_puzzle_with_solution()

    print("Test Puzzle:")
    print(puzzle_state)
    # Handle case where true_solution is empty (puzzle is solved)
    true_first_move = true_solution[0] if true_solution else 'None'
    print(f"True first move: {true_first_move}")

    # Calculate Manhattan distance for this puzzle
    manhattan_dist = test_puzzle.manhattan_distance(puzzle_state) # Pass the actual puzzle state
    print(f"Manhattan distance: {manhattan_dist}")

    # Prepare input: puzzle + manhattan distance (same as training data)
    puzzle_flat = puzzle_state.flatten()  # 9 numbers
    puzzle_input = np.append(puzzle_flat, manhattan_dist)  # Add manhattan distance
    puzzle_input = puzzle_input.reshape(1, -1)  # Reshape to (1, 10) for model

    # Get model prediction
    prediction = model.predict(puzzle_input, verbose=0)

    # Convert prediction to move
    predicted_move_num = prediction[0].argmax()  # Get highest probability
    predicted_move = encoder.inverse_transform([predicted_move_num])[0]
    confidence = prediction[0][predicted_move_num]

    print(f"Model prediction: {predicted_move} (confidence: {confidence:.2%})")

    # Show all move probabilities (optional - helpful for debugging)
    print("\nAll move probabilities:")
    # Iterate through all possible classes in the encoder
    for i in range(len(encoder.classes_)):
        move_name = encoder.inverse_transform([i])[0]
        print(f"  {move_name}: {prediction[0][i]:.1%}")

def debug_training_data(X, Y, encoder, num_examples=5):
    print("Debugging training data:")
    for i in range(num_examples):
        puzzle = X[i][:9].reshape(3,3)  # First 9 elements are puzzle
        manhattan = X[i][9]  # 10th element is manhattan distance
        move = encoder.inverse_transform([Y[i]])[0]
        
        print(f"\nExample {i+1}:")
        print("Puzzle:")
        print(puzzle)
        print(f"Manhattan distance: {manhattan}")
        print(f"Recommended move: {move}")

if __name__ == "__main__":
        X, Y, Z = generate_training_data(10000,3)
        X_prepared, Y_prepared, encoder = prepare_training_data(X, Y, Z)
        print("Training data prepared successfully.")

        model = create_puzzle_model()
        print("Model created successfully.")

        print("Training model...")
        x_train, x_test, y_train, y_test = train_test_split(X_prepared, Y_prepared, test_size=0.2, random_state=42)
        
        debug_training_data(X_prepared,Y_prepared,encoder)
        
        history = model.fit(
            x_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(x_test, y_test)
        )
        print("Model training completed.")

        # Test the trained model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        print(f"\nModel Performance:")
        print(f"Test Accuracy: {test_accuracy:.2%}")
        print(f"Test Loss: {test_loss:.3f}")

        # Test the model
        test_model_prediction(model, encoder)
