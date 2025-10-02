import numpy as np
import random
import copy
import math
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class SlidPuzzleTraingData():
    """"here we tried to exclude redundent moves such as "Up, Down, Up, Down" we could ignoor UP in the first place resulting "Up,Down,Left"."""
    
    def __init__(self, size=3,size_of_training_data=1000):
        self.size = size
        self.goal_state = self.generate_goal_state()
        self.training_data = self.generate_training_data(size_of_training_data)
        self.balanced_data = self.balance_data(self.training_data)
        
    def generate_goal_state(self):
        goal = np.arange(1, self.size * self.size + 1).reshape(self.size, self.size)
        goal[-1, -1] = 0
        return goal
    
    def generate_training_data(self, size_of_training_data):
        puzzle = self.goal_state.copy()
        training_data = []
        iteration = 0
        
        
        
        while len(training_data) < size_of_training_data:
            iteration+=1
            print("Iteration: ", iteration, end="\r")
            data = []
            valid_moves = []
            manhattan_distance = None
            
            empty_row, empty_col = np.where(puzzle == 0)
            empty_row, empty_col = empty_row[0], empty_col[0]
            
            #finds valid moves relative to the current possition
            if empty_row > 0: valid_moves.append('up')
            if empty_row < self.size - 1: valid_moves.append('down')  
            if empty_col > 0: valid_moves.append('left')
            if empty_col < self.size - 1: valid_moves.append('right')
            

            move = random.choice(valid_moves)
            #print(forDataCount, move)
            #moves the puzzle and reverse the move to make the target move
            if move == 'up':
                puzzle[empty_row, empty_col], puzzle[empty_row - 1, empty_col] = puzzle[empty_row - 1, empty_col], puzzle[empty_row, empty_col]
            elif move == 'down':
                puzzle[empty_row, empty_col], puzzle[empty_row + 1, empty_col] = puzzle[empty_row + 1, empty_col], puzzle[empty_row, empty_col]
            elif move == 'left':
                puzzle[empty_row, empty_col], puzzle[empty_row, empty_col - 1] = puzzle[empty_row, empty_col - 1], puzzle[empty_row, empty_col]
            elif move == 'right':
                puzzle[empty_row, empty_col], puzzle[empty_row, empty_col + 1] = puzzle[empty_row, empty_col + 1], puzzle[empty_row, empty_col]


            #finds the optimal move using bfs
            optimal_moves = self.optimalMoveBFS(puzzle)
            if optimal_moves:
                optimal_move = optimal_moves[0]
            else:
                optimal_move = None

            data.append(puzzle.copy().flatten())
            data.append(optimal_move)
            
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

    def optimalMoveBFS(self, puzzleState):
        goal = self.goal_state
        size = self.size
        start = tuple(map(tuple, puzzleState))
        goal = tuple(map(tuple, goal))

        
        if start == goal:
            return []

        queue = [(start, [])]
        visited = set()
        visited.add(start)
        #count=0

        while queue:
            #count+=1
            current_state, path = queue.pop(0)
            empty_row, empty_col = next((r, c) for r in range(size) for c in range(size) if current_state[r][c] == 0)

            directions = {
                'up': (empty_row - 1, empty_col),
                'down': (empty_row + 1, empty_col),
                'left': (empty_row, empty_col - 1),
                'right': (empty_row, empty_col + 1)
            }

            for move, (new_row, new_col) in directions.items():
                if 0 <= new_row < size and 0 <= new_col < size:
                    new_state = [list(row) for row in current_state]
                    new_state[empty_row][empty_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[empty_row][empty_col]
                    new_state_tuple = tuple(map(tuple, new_state))
                    #if count==100:
                        #print("Found solution:", path + [move])
                    if new_state_tuple not in visited:
                        if new_state_tuple == goal:
                            return path + [move]
                        visited.add(new_state_tuple)
                        queue.append((new_state_tuple, path + [move]))

        return None  # No solution found

    def balance_data(self, data):
        # balance the data and saving the data into a csv file
        move_counts = {"up": 0, "down": 0, "left": 0, "right": 0}
        for item in data:
            move = item[1]
            # Extract the move from numpy array if needed
            if isinstance(move, np.ndarray):
                move = move.item()  # or move[0] if it's a 1D array
            
            if move in move_counts:
                move_counts[move] += 1

        max_count = max(move_counts.values())
        balanced_data = []
        for move, count in move_counts.items():
            if count < max_count:
                needed = max_count - count
                # Also handle the move extraction in the filtering
                samples = [item for item in data 
                        if (item[1].item() if isinstance(item[1], np.ndarray) else item[1]) == move]
                if samples:
                    balanced_data.extend(random.choices(samples, k=needed))

        balanced_data.extend(data)
        random.shuffle(balanced_data)
        with open("balanced_training_data.csv", "w") as f:
            f.write("puzzle_state,target_move,manhattan_distance\n")
            for item in balanced_data:
                puzzle_state = ' '.join(map(str, item[0]))
                target_move = item[1].item() if isinstance(item[1], np.ndarray) else item[1]
                manhattan_distance = item[2]
                f.write(f"{puzzle_state},{target_move},{manhattan_distance}\n")
        return balanced_data
    
if __name__=="__main__":
    puzle = SlidPuzzleTraingData(3,5000)
    l,r,d,u,n = 0,0,0,0,0
    for i in puzle.balanced_data:
        if i[1]=='left':
            l+=1
        elif i[1]=='right':
            r+=1
        elif i[1]=='down':
            d+=1
        elif i[1]=='up':
            u+=1
        else:
            n+=1
            
    print(f"Left: {l}, Right: {r}, Down: {d}, Up: {u}, None: {n}")