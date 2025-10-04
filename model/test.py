import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load model
model = load_model("puzzle_solver_ann(My).keras")

# Your puzzle state
puzzle = [1,2,3,4,5,6,7,0,8]

# One-hot encoding
one_hot = np.eye(9)[puzzle].flatten().reshape(1, -1)  # shape = (1,81)

pred_probs = model.predict(one_hot)        # softmax probabilities
predicted_move_index = np.argmax(pred_probs)  
print(pred_probs)

# Map back to move string
move_map = {0:'up', 1:'down', 2:'left', 3:'right'}
predicted_move = move_map[predicted_move_index]

print("Predicted Move:", predicted_move)