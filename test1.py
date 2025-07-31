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
def generate_training_data(num_samples = 100, size = 3):
    """Generate random sliding puzzle states for testing."""
    print(f"Generating {num_samples} of training samples for {size}x{size} sliding puzzle...")
    
    puzzles = []
    solutions = []
    
    puzzle_gen = SlidingPuzzle(size)
    
    for i in range(num_samples):
        
        if i % 1000 == 0:
            print(f"Generated {i}/{num_samples} samples...")
        
        
        puzzle_gen.current_state, solution = puzzle_gen.create_puzzle_with_solution()
        puzzles.append(puzzle_gen.current_state.flatten())
        if len(solution) > 0:
            solutions.append(solution[0])  # Just first move
        else:
            solutions.append('no_move') 
    
    puzzles = np.array(puzzles)
    solutions = np.array(solutions)
    print(f"Generated {num_samples} samples for {size}x{size} sliding puzzle.")
    print(f"Sample puzzle: {puzzles[0]}")
    print(f"Sample solution: {solutions[:5]}... (truncated)\n")
    
    return np.array(puzzles), solutions
    
def prepare_training_data(x,y):
    """Prepare training data for the sliding puzzle."""
    
    # Convert puzzles to numpy arrays
    x = np.array(x)
    y = np.array(y)

    #convert move strings to integers
    move_encoder = LabelEncoder()
    move_encoder.fit(['up', 'down', 'left', 'right', 'no_move'])

    # Convert moves to integers
    y_encoded = move_encoder.fit_transform(y)

    print("Moves Encoding:")

    for i,move in enumerate(['up', 'down', 'left', 'right', 'no_move']):
        print(f"    {move} -> {i}")

    print(f"\nData ready for encoding")
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y_encoded.shape}\n")

    return x, y_encoded, move_encoder
    
def create_puzzle_model():
    """Create a simple NN for puzzle solving."""
    
    model =models.Sequential([
        #input layer
        layers.Dense(64, activation='relu', input_shape=(9,)),  # 3x3 puzzle flattened

        #hidden layer
        layers.Dense(64, activation='relu'),

        #output layer
        layers.Dense(5, activation='softmax')  # 5 possible moves
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
        print(f"True first move: {true_solution[0] if true_solution else 'None'}")
        
        # Get model prediction
        puzzle_input = puzzle_state.flatten().reshape(1, -1)  # Reshape for model
        prediction = model.predict(puzzle_input, verbose=0)
        
        # Convert prediction to move
        predicted_move_num = prediction[0].argmax()  # Get highest probability
        predicted_move = encoder.inverse_transform([predicted_move_num])[0]
        confidence = prediction[0][predicted_move_num]
        
        print(f"Model prediction: {predicted_move} (confidence: {confidence:.2%})")

    

if __name__ == "__main__":
        X, Y = generate_training_data(10000,3)
        X_prepared, Y_prepared, encoder = prepare_training_data(X, Y)
        print("Training data prepared successfully.")

        model = create_puzzle_model()
        print("Model created successfully.")

        print("Training model...")
        x_train, x_test, y_train, y_test = train_test_split(X_prepared, Y_prepared, test_size=0.2, random_state=42)
        
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

        # Look at first few examples
        # print("First 5 training examples:")
        # for i in range(5):
        #     puzzle = X_prepared[i]
        #     move_num = Y_prepared[i]
        #     move_name = encoder.inverse_transform([move_num])[0]
        #     print(f"Puzzle: {puzzle} -> Move: {move_name}")