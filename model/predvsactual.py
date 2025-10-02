import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load model
model = load_model("puzzle_solver_ann.keras")

# Rebuild X and y from your CSV (like you did for training)
training_data = pd.read_csv('balanced_training_data50KdeepSeek.csv')
training_data = training_data.dropna(subset=['target_move'])

training_data['puzzle_state'] = training_data['puzzle_state'].apply(
    lambda x: np.eye(9)[np.array(x.split(' ')).astype(int)].flatten()
)

print(training_data['puzzle_state'][1])

# Encode target moves
move_map = {'up':0, 'down':1, 'left':2, 'right':3}
inv_move_map = {v:k for k,v in move_map.items()}
y_true = training_data['target_move'].map(move_map).values

X = np.stack(training_data['puzzle_state'].values)

# Predictions
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert back from one-hot encoding to the actual puzzle numbers
def decode_puzzle_state(one_hot_vector):
    # one_hot_vector shape = (81,)
    one_hot_matrix = one_hot_vector.reshape(9, 9)  # 9 tiles × 9 categories
    return one_hot_matrix.argmax(axis=1)  # gives you the original 9 numbers

comparison_df = pd.DataFrame({
    "Puzzle_State": [' '.join(map(str, decode_puzzle_state(state))) for state in X],
    "Actual_Move": [inv_move_map[val] for val in y_true],
    "Predicted_Move": [inv_move_map[val] for val in y_pred]
})

print(comparison_df.head(20))

#stats: accuracy

accuracy = np.mean(y_true == y_pred)
print(f"Overall Accuracy: {accuracy:.4f}")
