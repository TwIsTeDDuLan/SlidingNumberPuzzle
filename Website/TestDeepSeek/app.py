from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model once when app starts
model = load_model("puzzle_solver_ann.keras")

# Move mapping
move_map = {0: 'down', 1: 'up', 2: 'right', 3: 'left'}

def predict_move(puzzle_state):
    """Predict the best move for a given puzzle state"""
    # Convert to one-hot encoding
    one_hot = np.eye(9)[puzzle_state].flatten().reshape(1, -1)
    
    # Make prediction
    pred_probs = model.predict(one_hot, verbose=0)
    predicted_move_index = np.argmax(pred_probs)
    
    return move_map[predicted_move_index], pred_probs[0].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        puzzle_state = data['puzzle']
        
        # Validate input
        if len(puzzle_state) != 9 or not all(isinstance(x, int) for x in puzzle_state):
            return jsonify({'error': 'Invalid puzzle state'}), 400
        
        predicted_move, probabilities = predict_move(puzzle_state)
        
        return jsonify({
            'predicted_move': predicted_move,
            'probabilities': {
                'up': probabilities[0],
                'down': probabilities[1],
                'left': probabilities[2],
                'right': probabilities[3]
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)