import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

training_data = pd.read_csv('balanced_training_data50KdeepSeekv2.csv')
training_data = training_data.dropna(subset=['target_move'])

training_data['puzzle_state'] = training_data['puzzle_state'].apply(lambda x: np.array(x.split(' ')).astype(np.int8))

training_data['puzzle_state'] = training_data['puzzle_state'].apply(
    lambda x: np.eye(9)[x].flatten().astype(np.int32)
)

# encode target moves (up:0, down:1, left:2, right:3)
training_data['target_move_encoded'] = training_data['target_move'].apply(lambda x: {'up':0, 'down':1, 'left':2, 'right':3,}[x])

INPUT_SIZE = 81
OUTPUT_SIZE = 4   # moves: up, down, left, right

# Build the model
model = Sequential([
    Input(shape=(INPUT_SIZE,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(OUTPUT_SIZE, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())