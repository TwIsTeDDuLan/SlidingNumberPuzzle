import numpy as np
import random
import collections
import time

class SlidPuzzleTraingData():
    def __init__(self, size=3, size_of_training_data=1000, use_precomputed=True):
        self.size = size
        self.goal_state = self.generate_goal_state()
        
        # Precompute optimal moves for common states to avoid BFS
        self.precomputed_moves = {}
        if use_precomputed:
            self.precompute_common_states()
            
        self.training_data = self.generate_training_data_optimized(size_of_training_data)
        self.balanced_data = self.balance_data(self.training_data)
    
    def precompute_common_states(self, max_depth=8):  # Increased depth
        """Precompute optimal moves for states close to goal"""
        print("Precomputing common states...")
        goal = tuple(self.goal_state.flatten())
        queue = collections.deque([(goal, [])])
        self.precomputed_moves[goal] = []
        
        states_computed = 0
        while queue and states_computed < 50000:  # Increased limit
            state, path = queue.popleft()
            if len(path) >= max_depth:
                continue
                
            state_array = np.array(state).reshape(self.size, self.size)
            empty_pos = np.where(state_array == 0)
            empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
            
            for move, (dr, dc) in [('up', (1, 0)), ('down', (-1, 0)), 
                                  ('left', (0, 1)), ('right', (0, -1))]:
                new_row, new_col = empty_row + dr, empty_col + dc
                
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    new_state = state_array.copy()
                    new_state[empty_row, empty_col], new_state[new_row, new_col] = \
                        new_state[new_row, new_col], new_state[empty_row, empty_col]
                    
                    new_state_tuple = tuple(new_state.flatten())
                    
                    if new_state_tuple not in self.precomputed_moves:
                        self.precomputed_moves[new_state_tuple] = [move] + path
                        queue.append((new_state_tuple, [move] + path))
                        states_computed += 1
        
        print(f"Precomputed {states_computed} states")
    
    def generate_training_data_optimized(self, size_of_training_data):
        """Optimized training data generation - FIXED to generate more data"""
        print("Generating training data...")
        training_data = []
        seen_states = set()
        
        # Use multiple generation strategies
        strategies = [
            self.generate_via_random_walks,
            self.generate_via_scrambling,
            self.generate_via_bfs_expansion
        ]
        
        samples_per_strategy = size_of_training_data // len(strategies) + 1000
        
        for strategy in strategies:
            if len(training_data) >= size_of_training_data:
                break
            new_samples = strategy(samples_per_strategy, seen_states)
            training_data.extend(new_samples)
            print(f"Strategy generated {len(new_samples)} samples, total: {len(training_data)}")
        
        # If still not enough, use simple random generation
        while len(training_data) < size_of_training_data:
            remaining = size_of_training_data - len(training_data)
            simple_samples = self.simple_random_generation(min(remaining * 2, 10000), seen_states)
            training_data.extend(simple_samples)
            print(f"Simple generation added {len(simple_samples)} samples, total: {len(training_data)}")
        
        return training_data[:size_of_training_data]
    
    def generate_via_random_walks(self, num_samples, seen_states):
        """Generate data via longer random walks from multiple start states"""
        samples = []
        start_states = self.generate_diverse_states(100)  # More start states
        
        for state in start_states:
            if len(samples) >= num_samples:
                break
            walk_samples = self.random_walk_from_state(
                state, 
                max_samples=num_samples // len(start_states) + 10,
                max_steps=20  # Longer walks
            )
            for puzzle_state, move, distance in walk_samples:
                state_bytes = puzzle_state.tobytes()
                if state_bytes not in seen_states:
                    samples.append([puzzle_state, move, distance])
                    seen_states.add(state_bytes)
        
        return samples
    
    def generate_via_scrambling(self, num_samples, seen_states):
        """Generate data by scrambling the puzzle with more moves"""
        samples = []
        
        for _ in range(num_samples * 2):  # Generate more to account for duplicates
            if len(samples) >= num_samples:
                break
            
            # Start from goal and make many random moves
            state = self.goal_state.copy()
            moves_made = []
            
            # Scramble with 10-25 moves to get more diverse states
            for _ in range(random.randint(15, 30)):
                empty_pos = np.where(state == 0)
                empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
                valid_moves = []
                
                if empty_row > 0 and (not moves_made or moves_made[-1] != 'down'): 
                    valid_moves.append('up')
                if empty_row < self.size - 1 and (not moves_made or moves_made[-1] != 'up'): 
                    valid_moves.append('down')
                if empty_col > 0 and (not moves_made or moves_made[-1] != 'right'): 
                    valid_moves.append('left')
                if empty_col < self.size - 1 and (not moves_made or moves_made[-1] != 'left'): 
                    valid_moves.append('right')
                
                if valid_moves:
                    move = random.choice(valid_moves)
                    self.apply_move(state, move)
                    moves_made.append(move)
            
            # Get optimal move and add to training data
            optimal_move = self.get_optimal_move_fast(state)
            if optimal_move:
                distance = self.fast_manhattan_distance(state)
                state_bytes = state.flatten().tobytes()
                if state_bytes not in seen_states:
                    samples.append([state.flatten(), optimal_move, distance])
                    seen_states.add(state_bytes)
        
        return samples
    
    def generate_via_bfs_expansion(self, num_samples, seen_states):
        """Generate data using BFS-like expansion from goal state"""
        samples = []
        queue = collections.deque([(self.goal_state.copy(), [])])
        
        while queue and len(samples) < num_samples:
            state, path = queue.popleft()
            
            # Add current state to samples if it has a path
            if path:
                optimal_move = path[0] if path else self.get_optimal_move_fast(state)
                if optimal_move:
                    distance = self.fast_manhattan_distance(state)
                    state_bytes = state.flatten().tobytes()
                    if state_bytes not in seen_states:
                        samples.append([state.flatten(), optimal_move, distance])
                        seen_states.add(state_bytes)
            
            # Expand to neighboring states (limit depth)
            if len(path) < 6:  # Limit depth to avoid explosion
                empty_pos = np.where(state == 0)
                empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
                
                for move, (dr, dc) in [('up', (1, 0)), ('down', (-1, 0)), 
                                      ('left', (0, 1)), ('right', (0, -1))]:
                    new_row, new_col = empty_row + dr, empty_col + dc
                    
                    if 0 <= new_row < self.size and 0 <= new_col < self.size:
                        new_state = state.copy()
                        self.apply_move(new_state, move)
                        
                        # Only add if not seen
                        new_state_bytes = new_state.flatten().tobytes()
                        if new_state_bytes not in seen_states:
                            queue.append((new_state, path + [move]))
                            seen_states.add(new_state_bytes)
        
        return samples
    
    def simple_random_generation(self, num_samples, seen_states):
        """Simple but effective random generation"""
        samples = []
        state = self.goal_state.copy()
        
        while len(samples) < num_samples:
            # Make a random move
            empty_pos = np.where(state == 0)
            empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
            valid_moves = []
            
            if empty_row > 0: valid_moves.append('up')
            if empty_row < self.size - 1: valid_moves.append('down')
            if empty_col > 0: valid_moves.append('left')
            if empty_col < self.size - 1: valid_moves.append('right')
            
            if valid_moves:
                move = random.choice(valid_moves)
                self.apply_move(state, move)
                
                # Add to training data
                optimal_move = self.get_optimal_move_fast(state)
                if optimal_move:
                    distance = self.fast_manhattan_distance(state)
                    state_bytes = state.flatten().tobytes()
                    if state_bytes not in seen_states:
                        samples.append([state.flatten(), optimal_move, distance])
                        seen_states.add(state_bytes)
            
            # Occasionally reset to avoid getting stuck
            if random.random() < 0.1:
                state = self.goal_state.copy()
        
        return samples

    # Keep your existing methods for generate_diverse_states, random_walk_from_state, 
    # get_optimal_move_fast, fast_manhattan_distance, apply_move, generate_goal_state, 
    # and balance_data - they should work fine
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
        with open("balanced_training_data50KdeepSeekv2.csv", "w") as f:
            f.write("puzzle_state,target_move,manhattan_distance\n")
            for item in balanced_data:
                puzzle_state = ' '.join(map(str, item[0]))
                target_move = item[1].item() if isinstance(item[1], np.ndarray) else item[1]
                manhattan_distance = item[2]
                f.write(f"{puzzle_state},{target_move},{manhattan_distance}\n")
        return balanced_data
    
    def get_optimal_move_fast(self, state):
        """Fast optimal move lookup using precomputed data and heuristics"""
        state_tuple = tuple(state.flatten())
        
        # First try precomputed moves
        if state_tuple in self.precomputed_moves:
            path = self.precomputed_moves[state_tuple]
            return path[0] if path else None
        
        # Fallback to greedy heuristic based on Manhattan distance improvement
        empty_pos = np.where(state == 0)
        empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
        
        best_move = None
        best_improvement = float('-inf')
        
        for move, (dr, dc) in [('up', (-1, 0)), ('down', (1, 0)), 
                              ('left', (0, -1)), ('right', (0, 1))]:
            new_row, new_col = empty_row + dr, empty_col + dc
            
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                # Calculate how much this move improves Manhattan distance
                tile_value = state[new_row, new_col]
                goal_pos = np.where(self.goal_state == tile_value)
                goal_row, goal_col = goal_pos[0][0], goal_pos[1][0]
                
                current_distance = abs(new_row - goal_row) + abs(new_col - goal_col)
                new_distance = abs(empty_row - goal_row) + abs(empty_col - goal_col)
                improvement = current_distance - new_distance
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = move
        
        return best_move
    
    def fast_manhattan_distance(self, state):
        """Optimized Manhattan distance calculation"""
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                val = state[i, j]
                if val != 0:
                    goal_i, goal_j = (val - 1) // self.size, (val - 1) % self.size
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

    def apply_move(self, state, move):
        """Apply a move to the puzzle state in-place"""
        empty_pos = np.where(state == 0)
        empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
        
        if move == 'up' and empty_row > 0:
            state[empty_row, empty_col], state[empty_row - 1, empty_col] = \
                state[empty_row - 1, empty_col], state[empty_row, empty_col]
        elif move == 'down' and empty_row < self.size - 1:
            state[empty_row, empty_col], state[empty_row + 1, empty_col] = \
                state[empty_row + 1, empty_col], state[empty_row, empty_col]
        elif move == 'left' and empty_col > 0:
            state[empty_row, empty_col], state[empty_row, empty_col - 1] = \
                state[empty_row, empty_col - 1], state[empty_row, empty_col]
        elif move == 'right' and empty_col < self.size - 1:
            state[empty_row, empty_col], state[empty_row, empty_col + 1] = \
                state[empty_row, empty_col + 1], state[empty_row, empty_col]
    
    
    def generate_goal_state(self):
        goal = np.arange(1, self.size * self.size + 1).reshape(self.size, self.size)
        goal[-1, -1] = 0
        return goal

    def generate_diverse_states(self, num_states):
        """Generate diverse starting states - FIXED to generate more states"""
        states = []
        base_state = self.goal_state.copy()
        
        # Use more patterns and more iterations
        patterns = [
            ['right', 'down', 'left', 'up'],
            ['down', 'right', 'up', 'left'],
            ['right', 'right', 'down', 'down'],
            ['up', 'left', 'down', 'right'],
            ['left', 'up', 'right', 'down'],
            ['down', 'left', 'up', 'right'],
        ]
        
        iterations_per_pattern = max(1, num_states // len(patterns))
        
        for pattern in patterns:
            for _ in range(iterations_per_pattern * 2):  # Generate more
                state = base_state.copy()
                moves_made = []
                
                # Apply more moves for greater diversity
                for i in range(random.randint(8, 20)):
                    empty_pos = np.where(state == 0)
                    empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
                    valid_moves = []
                    
                    if empty_row > 0 and (not moves_made or moves_made[-1] != 'down'): 
                        valid_moves.append('up')
                    if empty_row < self.size - 1 and (not moves_made or moves_made[-1] != 'up'): 
                        valid_moves.append('down')
                    if empty_col > 0 and (not moves_made or moves_made[-1] != 'right'): 
                        valid_moves.append('left')
                    if empty_col < self.size - 1 and (not moves_made or moves_made[-1] != 'left'): 
                        valid_moves.append('right')
                    
                    if valid_moves:
                        if random.random() < 0.6 and pattern[i % len(pattern)] in valid_moves:
                            move = pattern[i % len(pattern)]
                        else:
                            move = random.choice(valid_moves)
                        
                        self.apply_move(state, move)
                        moves_made.append(move)
                
                state_bytes = state.tobytes()
                if state_bytes not in [s.tobytes() for s in states]:
                    states.append(state.copy())
                    if len(states) >= num_states:
                        return states
        
        return states

    def random_walk_from_state(self, start_state, max_samples=20, max_steps=15):  # Increased limits
        """Generate samples from a starting state via random walk - FIXED to generate more"""
        samples = []
        current_state = start_state.copy()
        previous_move = None
        
        for step in range(max_steps):
            empty_pos = np.where(current_state == 0)
            empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
            valid_moves = []
            
            # Avoid reversing the previous move
            if empty_row > 0 and previous_move != 'down': valid_moves.append('up')
            if empty_row < self.size - 1 and previous_move != 'up': valid_moves.append('down')
            if empty_col > 0 and previous_move != 'right': valid_moves.append('left')
            if empty_col < self.size - 1 and previous_move != 'left': valid_moves.append('right')
            
            if not valid_moves:
                break
                
            move = random.choice(valid_moves)
            next_state = current_state.copy()
            self.apply_move(next_state, move)
            
            # Get optimal move
            optimal_move = self.get_optimal_move_fast(next_state)
            
            if optimal_move:
                manhattan_distance = self.fast_manhattan_distance(next_state)
                samples.append((next_state.flatten(), optimal_move, manhattan_distance))
            
            current_state = next_state
            previous_move = move
            
            if len(samples) >= max_samples:
                break
        
        return samples

if __name__ == "__main__":
    start_time = time.time()
    puzzle = SlidPuzzleTraingData(3, 100000, use_precomputed=True)  # Now should generate 50K
    end_time = time.time()
    
    print(f"Data generation took {end_time - start_time:.2f} seconds")
    
    l, r, d, u, n = 0, 0, 0, 0, 0
    for i in puzzle.balanced_data:
        if i[1] == 'left':
            l += 1
        elif i[1] == 'right':
            r += 1
        elif i[1] == 'down':
            d += 1
        elif i[1] == 'up':
            u += 1
        else:
            n += 1
            
    print(f"Left: {l}, Right: {r}, Down: {d}, Up: {u}, None: {n}")
    print(f"Total samples: {len(puzzle.balanced_data)}")