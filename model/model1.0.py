import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

training_data = pd.read_csv('balanced_training_data50KdeepSeek.csv')
training_data = training_data.dropna(subset=['target_move'])

training_data['puzzle_state'] = training_data['puzzle_state'].apply(lambda x: np.array(x.split(' ')).astype(np.int8))

training_data['puzzle_state'] = training_data['puzzle_state'].apply(
    lambda x: np.eye(9)[x].flatten().astype(np.int32)
)

# encode target moves (up:0, down:1, left:2, right:3)
training_data['target_move_encoded'] = training_data['target_move'].apply(lambda x: {'up':0, 'down':1, 'left':2, 'right':3,}[x])
#the model

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

# ---- Example Training Data ----
# X: one-hot encoded puzzle states (num_samples, INPUT_SIZE)
# y: labels (0=up,1=down,2=left,3=right)
# In practice, generate this from your puzzle solver (A*/BFS)

X = np.stack(training_data['puzzle_state'].values)  # This creates shape (n_samples, 81)
y = training_data['target_move_encoded'].values


# Convert labels to one-hot
y_cat = to_categorical(y, num_classes=OUTPUT_SIZE)

# Split before training
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Train only on train data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")


# Save model
#model.save("puzzle_solver_ann.keras")

from sklearn.metrics import confusion_matrix, classification_report

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=['up','down','left','right']))
