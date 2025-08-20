import numpy as np
import random
import copy
import math
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class SlidPuzzleTraingData():
    """Approach 2 aims to provide puzzle state, manhattan distance, and target move to the model"""
    
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
            
            #choses a random move
            move = random.choice(valid_moves)
            
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


def prepare_training_data(training_data):
    # This function will prepare the training data for the model
    x_prepared = []
    y_prepared = []
    y=[]
    
    for data in training_data:
        x = np.hstack((data[0], data[2]))  # Puzzle state and Manhattan distance
        x_prepared.append(x)  # Append the prepared state
        y.append(data[1])  # Target move

    x_prepared = np.array(x_prepared)  
    y = np.array(y)  # Convert target moves to numpy array
    
    target_moves_encoder = LabelEncoder()
    target_moves_encoder.fit(['up', 'down', 'left', 'right'])
    y_prepared = target_moves_encoder.transform(y)  # Encode target moves 
    
    print("Prepared training data.")
    for i,move in enumerate(['up', 'down', 'left', 'right']):
        print(f'{move} -> {i}')
    
    print(f"X_combined shape: {x_prepared.shape}")  # Should be (num_samples, 10)
    print(f"Y_encoded shape: {y_prepared.shape}")
    
    return x_prepared, y_prepared, target_moves_encoder


def create_puzzle_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    puzzle = SlidPuzzleTraingData(3,1000)
    x , y, encoder = prepare_training_data(puzzle.training_data)

    model = create_puzzle_model()
    print("Model created successfully.")
    
    print("Training model...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        batch_size=32
    )
    print("Model training completed.")
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    
    print(f"\nModel Performance:")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Loss: {test_loss:.3f}")
          
          
    # print("Training Data:")
    # for data in puzzle.training_data:
    #      print(data[0])
    #      print("Target Move:", data[1])
    #      print("Manhattan Distance:", data[2])
    
    